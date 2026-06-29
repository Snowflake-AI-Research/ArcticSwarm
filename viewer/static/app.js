// Question List Page

let questionsData = [];
let sortCol = 'conv_id';
let sortAsc = true;
let filterCorrect = 'all';
let filterSearch = '';

function formatDuration(s) {
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function formatTokens(n) {
  if (!n) return '-';
  if (n < 1000) return String(n);
  if (n < 1000000) return `${(n / 1000).toFixed(0)}K`;
  return `${(n / 1000000).toFixed(1)}M`;
}

// Token cell that distinguishes "errored / no usage captured" from a genuine
// zero.  A bare "-" made errored/timed-out cases (config-build failure,
// connection crash, wall-timeout — which legitimately have no token usage)
// look identical to a clean run, so ~half the table read as empty.  Show
// "err"/"timeout" (with the error string on hover) instead.
function tokenCell(q) {
  const t = q.total_tokens;
  if (!t) {
    if (q.has_error || q.had_timeout) {
      const reason = q.error ? String(q.error) : (q.had_timeout ? 'timed out' : 'errored');
      const title = reason.replace(/"/g, '&quot;').slice(0, 300);
      const label = q.had_timeout ? 'timeout' : 'err';
      return `<span class="td-error" title="${title}">${label}</span>`;
    }
    return '-';
  }
  return formatTokens(t);
}

function getFiltered() {
  let data = [...questionsData];
  if (filterCorrect === 'correct') data = data.filter(q => q.judge_correct === true);
  else if (filterCorrect === 'incorrect') data = data.filter(q => q.judge_correct === false);
  else if (filterCorrect === 'error') data = data.filter(q => q.has_error);
  else if (filterCorrect === 'timeout') data = data.filter(q => q.had_timeout);

  if (filterSearch) {
    const s = filterSearch.toLowerCase();
    data = data.filter(q => q.conv_id.toLowerCase().includes(s) || q.question.toLowerCase().includes(s));
  }

  data.sort((a, b) => {
    let va = a[sortCol], vb = b[sortCol];
    if (va == null) va = '';
    if (vb == null) vb = '';
    if (typeof va === 'string') {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return sortAsc ? va - vb : vb - va;
  });
  return data;
}

function renderTable(container) {
  const data = getFiltered();
  const cols = [
    { key: 'conv_id', label: 'Question' },
    { key: 'duration_seconds', label: 'Latency' },
    { key: 'total_tokens', label: 'Tokens' },
    { key: 'judge_correct', label: 'Correct' },
    { key: 'reference_answer', label: 'Answer' },
    { key: 'has_error', label: 'Error' },
    { key: 'had_timeout', label: 'Timeout' },
    { key: 'swarm_bbs_message_count', label: 'BBS Msgs' },
    { key: 'swarm_teammates_spawned', label: 'Agents' },
  ];

  let html = '<table><thead><tr>';
  for (const col of cols) {
    const cls = sortCol === col.key ? 'sorted' : '';
    const arrow = sortCol === col.key ? (sortAsc ? ' ▲' : ' ▼') : '';
    html += `<th class="${cls}" data-col="${col.key}">${col.label}${arrow}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (const q of data) {
    const correctCls = q.judge_correct === true ? 'badge-correct' : q.judge_correct === false ? 'badge-incorrect' : '';
    const correctText = q.judge_correct === true ? 'Yes' : q.judge_correct === false ? 'No' : '-';
    const errorText = q.has_error ? `<span class="td-error">Yes</span>` : '<span class="td-muted">-</span>';
    const timeoutText = q.had_timeout ? `<span class="td-timeout">Yes</span>` : '<span class="td-muted">-</span>';
    const answerText = q.reference_answer || '-';
    const answerTrunc = answerText.length > 30 ? answerText.slice(0, 30) + '...' : answerText;

    html += `<tr data-id="${q.conv_id}">
      <td class="td-mono">${q.conv_id}</td>
      <td>${formatDuration(q.duration_seconds)}</td>
      <td>${tokenCell(q)}</td>
      <td><span class="badge ${correctCls}">${correctText}</span></td>
      <td class="td-muted" title="${answerText}">${answerTrunc}</td>
      <td>${errorText}</td>
      <td>${timeoutText}</td>
      <td>${q.swarm_bbs_message_count}</td>
      <td>${q.swarm_teammates_spawned}</td>
    </tr>`;
  }
  html += '</tbody></table>';

  container.innerHTML = html;

  // Sort handlers
  container.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortCol === col) sortAsc = !sortAsc;
      else { sortCol = col; sortAsc = true; }
      renderTable(container);
    });
  });

  // Click handlers
  container.querySelectorAll('tr[data-id]').forEach(tr => {
    tr.addEventListener('click', () => {
      window.location.hash = `#/question/${encodeURIComponent(tr.dataset.id)}`;
    });
  });
}

export async function renderQuestionList(app) {
  app.innerHTML = '<div class="loading">Loading questions</div>';

  const res = await fetch('/api/questions');
  questionsData = await res.json();

  const total = questionsData.length;
  const correct = questionsData.filter(q => q.judge_correct === true).length;
  const errors = questionsData.filter(q => q.has_error).length;
  const accuracy = total > 0 ? (correct / total * 100).toFixed(1) : '0';

  app.innerHTML = `
    <div class="header">
      <h1>Arcticswarm BBS Viewer</h1>
      <div class="stats">
        <span>Accuracy: <span class="stat-value">${accuracy}%</span></span>
        <span>Questions: <span class="stat-value">${total}</span></span>
        <span>Errors: <span class="stat-value">${errors}</span></span>
      </div>
    </div>
    <div class="filters">
      <select id="filter-correct">
        <option value="all">All</option>
        <option value="correct">Correct only</option>
        <option value="incorrect">Incorrect only</option>
        <option value="error">Errors only</option>
        <option value="timeout">Timeouts only</option>
      </select>
      <input type="text" id="filter-search" placeholder="Search question or conv_id...">
    </div>
    <div id="table-container"></div>
  `;

  const tableContainer = document.getElementById('table-container');

  document.getElementById('filter-correct').addEventListener('change', e => {
    filterCorrect = e.target.value;
    renderTable(tableContainer);
  });
  document.getElementById('filter-search').addEventListener('input', e => {
    filterSearch = e.target.value;
    renderTable(tableContainer);
  });

  renderTable(tableContainer);
}
