// Question Detail Page - BBS Board Replay

const CHANNEL_ORDER = [
  'tasks', 'discoveries', 'key-findings',
  'work-logs',
  'discussion', 'consensus'
];

const CHANNEL_CLASSES = {
  'tasks': 'ch-tasks',
  'discoveries': 'ch-discoveries',
  'key-findings': 'ch-key-findings',
  'work-logs': 'ch-work-logs',
  'consensus': 'ch-consensus',
  'discussion': 'ch-discussion',
};

function formatTime(ts) {
  if (!ts) return '';
  try {
    const d = new Date(ts);
    return d.toISOString().slice(11, 19);
  } catch { return ''; }
}

function formatDuration(s) {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function formatTokens(n) {
  if (!n || n === 0) return '-';
  if (n < 1000) return String(n);
  if (n < 1000000) return `${(n / 1000).toFixed(0)}K`;
  return `${(n / 1000000).toFixed(1)}M`;
}

// Token label that flags errored/timed-out cases (which legitimately captured
// no usage) instead of showing a bare "-" that looks like a clean zero.
function tokenLabel(data) {
  if (!data.total_tokens) {
    if (data.has_error || data.had_timeout) {
      return data.had_timeout ? 'errored (timeout)' : 'errored';
    }
    return '- tokens';
  }
  return `${formatTokens(data.total_tokens)} tokens`;
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMarkdown(text) {
  if (typeof marked !== 'undefined' && marked.parse) {
    try { return marked.parse(text); } catch {}
  }
  return `<pre>${escapeHtml(text)}</pre>`;
}

function truncate(text, max = 200) {
  if (!text) return '';
  if (text.length <= max) return text;
  return text.slice(0, max) + '...';
}

// Find the latest BBS snapshot at or before the given step
function findBBSSnapshot(snapshots, bbsPostSteps, step) {
  let bestStep = -1;
  for (const s of bbsPostSteps) {
    if (s <= step) bestStep = s;
    else break;
  }
  if (bestStep >= 0 && snapshots[String(bestStep)]) {
    return snapshots[String(bestStep)];
  }
  return null;
}

// Determine agent states at a given step
function computeAgentStates(events, agents, step) {
  const states = {};
  for (const agent of agents) {
    states[agent.name] = { status: 'not-spawned', detail: '', fullText: '', tool: '' };
  }
  states['orchestrator'] = { status: 'idle', detail: '', fullText: '', tool: '' };

  for (let i = 0; i <= step && i < events.length; i++) {
    const e = events[i];
    const actor = e.actor;

    if (e.event_type === 'spawn') {
      const name = e.text?.match(/Spawned (\w+)/)?.[1];
      if (name && states[name]) {
        states[name] = { status: 'idle', detail: 'Just spawned', fullText: e.text || '', tool: '' };
      }
    } else if (e.event_type === 'text') {
      states[actor] = { status: 'thinking', detail: truncate(e.text, 60), fullText: e.text || '', tool: '' };
    } else if (e.event_type === 'tool_call' || e.event_type === 'bbs_post' || e.event_type === 'task_create') {
      const toolName = e.tool_name || e.event_type;
      const fullInput = e.tool_input ? JSON.stringify(e.tool_input, null, 2) : (e.tool_input_summary || '');
      states[actor] = { status: 'working', detail: e.tool_input_summary || '', fullText: fullInput, tool: toolName };
    } else if (e.event_type === 'tool_result' || e.event_type === 'bbs_read') {
      const fullResult = e.tool_result_text || e.tool_result_summary || '';
      states[actor] = { status: 'thinking', detail: `Got result from ${e.tool_name || '?'}`, fullText: fullResult, tool: '' };
    }
  }
  return states;
}

// Render the BBS board panel
function renderBBSBoard(snapshot, currentStep) {
  if (!snapshot) {
    return '<div class="panel-body"><div class="td-muted" style="padding:12px;">No BBS posts yet</div></div>';
  }

  let html = `<div class="bbs-token-count">Board tokens: <span class="count">~${snapshot.total_estimated_tokens.toLocaleString()}</span> (${snapshot.posts.length} posts)</div>`;
  html += '<div class="panel-body">';

  const byChannel = snapshot.posts_by_channel || {};
  const channels = CHANNEL_ORDER.filter(c => byChannel[c]?.length > 0);
  for (const c of Object.keys(byChannel)) {
    if (!channels.includes(c) && byChannel[c]?.length > 0) channels.push(c);
  }

  for (const channel of channels) {
    const posts = byChannel[channel] || [];
    const cls = CHANNEL_CLASSES[channel] || '';
    html += `<div class="bbs-channel">`;
    html += `<div class="bbs-channel-header ${cls}" onclick="this.parentElement.classList.toggle('collapsed')">#${channel} <span class="count">(${posts.length})</span></div>`;
    html += `<div class="bbs-channel-posts">`;

    for (let i = 0; i < posts.length; i++) {
      const post = posts[i];
      const isNew = (i === posts.length - 1) && snapshot.step === currentStep;
      const highlightCls = isNew ? 'highlighted' : '';
      const timeStr = formatTime(post.timestamp);
      const contentHtml = renderMarkdown(post.content);

      html += `<div class="bbs-post ${highlightCls}">
        <span class="post-author">${escapeHtml(post.author)}</span>
        <span class="post-time">${timeStr}</span>
        <div class="post-content" onclick="this.classList.toggle('expanded')">${contentHtml}</div>
      </div>`;
    }
    html += '</div></div>';
  }
  html += '</div>';
  return html;
}

function renderAgentActivity(agentStates, agents) {
  let html = '';
  const orchState = agentStates['orchestrator'] || { status: 'idle', detail: '', fullText: '', tool: '' };
  html += renderAgentRow('orchestrator', orchState, null, false);
  for (const agent of agents) {
    const state = agentStates[agent.name] || { status: 'not-spawned', detail: '', fullText: '', tool: '' };
    html += renderAgentRow(agent.name, state, agent.profile, agent.is_idle_reviewer);
  }
  return html;
}

function renderAgentRow(name, state, profile, isIdleReviewer) {
  const statusClass = `status-${state.status === 'not-spawned' ? 'not-spawned' : state.status}`;
  const profileTag = profile ? `<span class="tool-tag">${profile}</span>` : '';
  const idleTag = isIdleReviewer ? '<span class="idle-reviewer-tag">idle reviewer</span>' : '';
  const toolTag = state.tool ? `<span class="tool-tag">${state.tool}</span>` : '';
  const statusIcon = state.status === 'thinking' ? '\u25cf ' :
                     state.status === 'working' ? '\u25b6 ' :
                     state.status === 'idle' ? '\u25cb ' : '';

  const hasFullText = state.fullText && state.fullText.length > 80;
  const expandIcon = hasFullText ? '<span class="expand-icon">\u25b8</span>' : '';
  const fullTextHtml = hasFullText
    ? `<div class="agent-full-text">${renderMarkdown(state.fullText)}</div>`
    : '';

  return `<div class="agent-row ${hasFullText ? 'expandable' : ''}" onclick="this.classList.toggle('expanded')">
    <span class="agent-name">${expandIcon}${escapeHtml(name)} ${idleTag}</span>
    <div class="agent-status ${statusClass}">
      ${profileTag}
      ${statusIcon}${toolTag}
      ${escapeHtml(truncate(state.detail, 80))}
    </div>
    ${fullTextHtml}
  </div>`;
}

function renderEventLog(events, currentStep) {
  let html = '';
  for (let i = 0; i < events.length; i++) {
    const e = events[i];
    const isCurrent = e.step === currentStep;
    const cls = isCurrent ? 'event-row current' : 'event-row';
    const typeCls = e.is_error ? 'event-error' : `event-type-${e.event_type}`;

    let summary = '';
    if (e.event_type === 'bbs_post') {
      summary = `#${e.bbs_channel}: ${truncate(e.bbs_content, 80)}`;
    } else if (e.event_type === 'tool_call' || e.event_type === 'task_create') {
      summary = e.tool_input_summary || '';
    } else if (e.event_type === 'tool_result' || e.event_type === 'bbs_read') {
      summary = truncate(e.tool_result_summary || '', 80);
    } else if (e.event_type === 'text') {
      summary = truncate(e.text, 80);
    } else if (e.event_type === 'spawn') {
      summary = e.text || '';
    }

    const typeLabel = e.event_type === 'tool_call' ? e.tool_name :
                      e.event_type === 'tool_result' ? `${e.tool_name} result` :
                      e.event_type;

    html += `<div class="${cls}" data-step="${e.step}">
      <span class="event-step">${e.step}</span>
      <span class="event-time">${formatTime(e.timestamp)}</span>
      <span class="event-actor">${escapeHtml(e.actor)}</span>
      <span class="event-type ${typeCls}">${escapeHtml(typeLabel)}</span>
      <span class="event-summary">${escapeHtml(summary)}</span>
    </div>`;
  }
  return html;
}

export async function renderDetail(app, convId) {
  app.innerHTML = '<div class="loading">Loading timeline</div>';

  const res = await fetch(`/api/questions/${encodeURIComponent(convId)}/timeline`);
  if (!res.ok) {
    app.innerHTML = `<div class="loading">Error loading timeline: ${res.status}</div>`;
    return;
  }

  const data = await res.json();
  const events = data.events || [];
  const bbsSnapshots = data.bbs_snapshots || {};
  const bbsPostSteps = data.bbs_post_steps || [];
  const agents = data.agents || [];
  const tasks = data.tasks || [];
  let currentStep = 0;
  let playing = false;
  let playInterval = null;
  let filterAgent = 'all';

  // Collect unique actors for the dropdown
  const actorSet = new Set();
  for (const e of events) { if (e.actor) actorSet.add(e.actor); }
  const actorList = ['all', ...['orchestrator', ...agents.map(a => a.name)].filter(a => actorSet.has(a))];

  function getFilteredEvents() {
    if (filterAgent === 'all') return events;
    return events.filter(e => e.actor === filterAgent);
  }

  function rerenderEventLog() {
    const eventLog = document.getElementById('event-log-body');
    if (!eventLog) return;
    const filtered = getFilteredEvents();
    eventLog.innerHTML = renderEventLog(filtered, currentStep);
    // Update count in title
    const title = document.getElementById('event-log-title');
    if (title) {
      const suffix = filterAgent !== 'all' ? ` — ${filterAgent}` : '';
      title.textContent = `Event Log (${filtered.length} events${suffix})`;
    }
  }

  function update() {
    const stepInfo = document.getElementById('step-info');
    if (stepInfo) {
      stepInfo.textContent = `Step ${currentStep} / ${events.length - 1}`;
      const timeInfo = document.getElementById('time-info');
      if (timeInfo && events[currentStep]) timeInfo.textContent = formatTime(events[currentStep].timestamp);
    }

    const slider = document.getElementById('timeline-slider');
    if (slider) slider.value = currentStep;

    const bbsPanel = document.getElementById('bbs-panel');
    if (bbsPanel) {
      const snapshot = findBBSSnapshot(bbsSnapshots, bbsPostSteps, currentStep);
      bbsPanel.innerHTML = '<div class="panel-title">BBS Board</div>' +
        renderBBSBoard(snapshot, currentStep);
    }

    const agentPanel = document.getElementById('agent-panel-body');
    if (agentPanel) {
      const states = computeAgentStates(events, agents, currentStep);
      agentPanel.innerHTML = renderAgentActivity(states, agents);
    }

    const eventLog = document.getElementById('event-log-body');
    if (eventLog) {
      // Find the row matching currentStep (may differ from index if filtered)
      const currentRow = eventLog.querySelector('.event-row.current');
      if (currentRow) currentRow.classList.remove('current');
      const target = eventLog.querySelector(`.event-row[data-step="${currentStep}"]`);
      if (target) {
        target.classList.add('current');
        target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }

  const durStr = formatDuration(data.total_duration_seconds || data.duration_seconds || 0);
  const maxStep = Math.max(0, events.length - 1);

  const correctBadge = data.judge_correct === true
    ? '<span class="badge badge-correct">Correct</span>'
    : data.judge_correct === false
      ? '<span class="badge badge-incorrect">Incorrect</span>'
      : '<span class="badge">Unknown</span>';

  const refAnswer = data.reference_answer || '(not available)';
  const judgeComment = data.judge_comment || '';
  const judgeRaw = data.judge_raw_output || '';
  const responsePreview = data.response_text ? truncate(data.response_text, 500) : '';

  app.innerHTML = `
    <div class="detail-header">
      <a class="back" href="#/">&larr; Back</a>
      <strong>${escapeHtml(convId)}</strong>
      ${correctBadge}
      <div class="meta">
        <span>${durStr}</span>
        <span>${tokenLabel(data)}</span>
        <span>${events.length} events</span>
        <span>${agents.length} agents</span>
        <span>${tasks.length} tasks</span>
      </div>
    </div>

    <div class="question-box">${escapeHtml(truncate(data.question, 1000))}</div>

    <div class="judge-section">
      <div class="judge-row">
        <span class="judge-label">Reference Answer:</span>
        <span class="judge-value ref-answer">${escapeHtml(refAnswer)}</span>
      </div>
      ${judgeComment ? `<div class="judge-row expandable-section" onclick="this.classList.toggle('expanded')">
        <span class="judge-label"><span class="expand-icon">\u25b8</span> Judge Comment:</span>
        <span class="judge-value judge-preview">${escapeHtml(truncate(judgeComment, 120))}</span>
        <div class="judge-full">${renderMarkdown(judgeComment)}</div>
      </div>` : ''}
      ${judgeRaw ? `<div class="judge-row expandable-section" onclick="this.classList.toggle('expanded')">
        <span class="judge-label"><span class="expand-icon">\u25b8</span> Judge Raw Output:</span>
        <span class="judge-value judge-preview">${escapeHtml(truncate(judgeRaw, 120))}</span>
        <div class="judge-full"><pre>${escapeHtml(judgeRaw)}</pre></div>
      </div>` : ''}
      ${responsePreview ? `<div class="judge-row expandable-section" onclick="this.classList.toggle('expanded')">
        <span class="judge-label"><span class="expand-icon">\u25b8</span> LLM Response:</span>
        <span class="judge-value judge-preview">${escapeHtml(truncate(responsePreview, 120))}</span>
        <div class="judge-full">${renderMarkdown(data.response_text)}</div>
      </div>` : ''}
    </div>

    <div class="timeline-bar">
      <div class="timeline-controls">
        <button id="btn-prev-bbs" title="Previous BBS post (Shift+Left)">&larr; Prev BBS</button>
        <button id="btn-prev" title="Previous step (Left)">&larr;</button>
        <button id="btn-play" title="Play/Pause (Space)">Play</button>
        <button id="btn-next" title="Next step (Right)">&rarr;</button>
        <button id="btn-next-bbs" title="Next BBS post (Shift+Right)">Next BBS &rarr;</button>
        <span id="step-info" class="step-info">Step 0 / ${maxStep}</span>
        <span id="time-info" class="time-info"></span>
      </div>
      <div class="slider-container">
        <div class="bbs-markers" id="bbs-markers"></div>
        <input type="range" id="timeline-slider" min="0" max="${maxStep}" value="0" step="1">
      </div>
    </div>

    <div class="panels">
      <div class="panel">
        <div class="panel-title">Agent Activity</div>
        <div class="panel-body" id="agent-panel-body"></div>
      </div>
      <div class="panel" id="bbs-panel"></div>
    </div>

    <div class="event-log">
      <div class="event-log-header">
        <span class="panel-title" id="event-log-title">Event Log (${events.length} events)</span>
        <select id="agent-filter" class="agent-filter">
          ${actorList.map(a => `<option value="${a}">${a === 'all' ? 'All agents' : a}</option>`).join('')}
        </select>
      </div>
      <div class="panel-body" id="event-log-body">
        ${renderEventLog(getFilteredEvents(), currentStep)}
      </div>
    </div>
  `;

  // Add BBS markers on slider
  const markersDiv = document.getElementById('bbs-markers');
  if (markersDiv && maxStep > 0) {
    for (const s of bbsPostSteps) {
      const pct = (s / maxStep) * 100;
      const marker = document.createElement('div');
      marker.className = 'bbs-marker';
      marker.style.left = `${pct}%`;
      markersDiv.appendChild(marker);
    }
  }

  // Event handlers
  const slider = document.getElementById('timeline-slider');
  slider.addEventListener('input', () => {
    currentStep = parseInt(slider.value);
    update();
  });

  document.getElementById('agent-filter').addEventListener('change', (e) => {
    filterAgent = e.target.value;
    rerenderEventLog();
  });

  document.getElementById('btn-prev').addEventListener('click', () => {
    if (currentStep > 0) { currentStep--; update(); }
  });

  document.getElementById('btn-next').addEventListener('click', () => {
    if (currentStep < maxStep) { currentStep++; update(); }
  });

  document.getElementById('btn-prev-bbs').addEventListener('click', () => {
    for (let i = bbsPostSteps.length - 1; i >= 0; i--) {
      if (bbsPostSteps[i] < currentStep) { currentStep = bbsPostSteps[i]; update(); return; }
    }
  });

  document.getElementById('btn-next-bbs').addEventListener('click', () => {
    for (const s of bbsPostSteps) {
      if (s > currentStep) { currentStep = s; update(); return; }
    }
  });

  document.getElementById('btn-play').addEventListener('click', () => {
    playing = !playing;
    document.getElementById('btn-play').textContent = playing ? 'Pause' : 'Play';
    if (playing) {
      playInterval = setInterval(() => {
        if (currentStep < maxStep) { currentStep++; update(); }
        else { playing = false; document.getElementById('btn-play').textContent = 'Play'; clearInterval(playInterval); }
      }, 200);
    } else {
      clearInterval(playInterval);
    }
  });

  // Click on event log row to jump
  document.getElementById('event-log-body').addEventListener('click', (e) => {
    const row = e.target.closest('.event-row');
    if (row) {
      currentStep = parseInt(row.dataset.step);
      update();
    }
  });

  // Keyboard shortcuts
  const keyHandler = (e) => {
    if (!window.location.hash.startsWith('#/question/')) {
      document.removeEventListener('keydown', keyHandler);
      return;
    }
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      if (e.shiftKey) document.getElementById('btn-prev-bbs')?.click();
      else document.getElementById('btn-prev')?.click();
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      if (e.shiftKey) document.getElementById('btn-next-bbs')?.click();
      else document.getElementById('btn-next')?.click();
    } else if (e.key === ' ') {
      e.preventDefault();
      document.getElementById('btn-play')?.click();
    }
  };
  document.addEventListener('keydown', keyHandler);

  update();
}
