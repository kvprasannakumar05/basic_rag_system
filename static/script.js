// API Base URL
const API_BASE = window.location.origin;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const documentsList = document.getElementById('documentsList');
const refreshBtn = document.getElementById('refreshBtn');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistory = document.getElementById('chatHistory');
const typingIndicator = document.getElementById('typingIndicator');


// Create Toast Container
const toast = document.createElement('div');
toast.className = 'toast';
document.body.appendChild(toast);

// === SESSION MANAGEMENT ===
let sessionId = localStorage.getItem('rag_session_id');
if (!sessionId) {
    sessionId = 'user_' + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('rag_session_id', sessionId);
}
console.log('Session ID:', sessionId);

// === UPLOAD LOGIC ===

uploadArea.addEventListener('click', (e) => {
    if (e.target.id !== 'browseLink') fileInput.click();
});

const browseLink = document.getElementById('browseLink');
if (browseLink) {
    browseLink.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
}

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
});

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#ffffff';
});
uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#333333';
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.style.borderColor = '#333333';
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
    const validTypes = ['.pdf', '.txt'];
    const fileName = file.name.toLowerCase();

    if (!validTypes.some(type => fileName.endsWith(type))) {
        showToast('Invalid file type (PDF/TXT only)', 'error');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showToast('File too large (>10MB)', 'error');
        return;
    }

    setUploadLoading(true);
    addGhostDocument(file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            headers: {
                'x-session-id': sessionId
            },
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showToast(`Indexed "${data.filename}"`, 'success');
            await loadDocuments();
            addBotMessage(`I've received "${data.filename}". I'm ready to answer questions about it.`);
        } else {
            showToast(`Error: ${data.detail}`, 'error');
            removeGhostDocument();
        }
    } catch (error) {
        showToast('Network error during upload', 'error');
        removeGhostDocument();
    } finally {
        setUploadLoading(false);
        fileInput.value = '';
    }
}

function setUploadLoading(isLoading) {
    const textP = uploadArea.querySelector('.upload-text');
    if (isLoading) {
        uploadArea.classList.add('upload-processing');
        textP.innerHTML = `<div class="spinner-mini"></div> Processing...`;
    } else {
        uploadArea.classList.remove('upload-processing');
        textP.innerHTML = `Drop files or <span class="browse-link" id="browseLink">Select</span>`;
        // Re-attach listener
        const newBrowse = document.getElementById('browseLink');
        if (newBrowse) {
            newBrowse.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.click();
            });
        }
    }
}

function addGhostDocument(filename) {
    if (documentsList.querySelector('.empty-state') || documentsList.innerText.includes('INDEX EMPTY')) {
        documentsList.innerHTML = '';
    }
    const ghost = document.createElement('div');
    ghost.className = 'document-card ghost';
    ghost.id = 'ghost-doc';
    ghost.innerHTML = `
        <div class="document-info">
            <div class="document-name">${escapeHtml(filename)}</div>
            <div class="document-meta">INDEXING...</div>
        </div>
    `;
    documentsList.insertBefore(ghost, documentsList.firstChild);
}

function removeGhostDocument() {
    const ghost = document.getElementById('ghost-doc');
    if (ghost) ghost.remove();
}


// === CHAT LOGIC ===

sendBtn.addEventListener('click', handleChat);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleChat();
    }
});
chatInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if (this.value === '') this.style.height = 'auto';
});

async function handleChat() {
    const message = chatInput.value.trim();
    if (!message) return;

    addUserMessage(message);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;
    showTyping();

    const requestData = {
        question: message,
        session_id: sessionId
    };

    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-session-id': sessionId
            },
            body: JSON.stringify(requestData)
        });
        const data = await response.json();

        hideTyping();
        if (response.ok) {
            addBotMessage(data.answer, data.sources);
        } else {
            addBotMessage(`Error: ${data.detail}`);
        }
    } catch (error) {
        hideTyping();
        addBotMessage(`Connection Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

function addUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    chatHistory.appendChild(div);
    scrollToBottom();
}

function addBotMessage(text, sources = []) {
    const div = document.createElement('div');
    div.className = 'message bot';
    let html = `<div class="message-content">${formatText(text)}`;

    if (sources && sources.length > 0) {
        html += `<div class="chat-sources"><div class="sources-title">REFERENCES</div>`;
        html += sources.map(s => `
            <span class="source-chip" title="${escapeHtml(s.chunk_text)}">
                ${escapeHtml(s.metadata.filename)}
            </span>
        `).join('');
        html += `</div>`;
    }
    html += `</div>`;
    div.innerHTML = html;
    chatHistory.appendChild(div);
    scrollToBottom();
}

function showTyping() { typingIndicator.style.display = 'block'; scrollToBottom(); }
function hideTyping() { typingIndicator.style.display = 'none'; }
function scrollToBottom() {
    chatHistory.scrollTo({
        top: chatHistory.scrollHeight,
        behavior: 'smooth'
    });
}

function formatText(text) {
    return escapeHtml(text)
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// === DOCUMENTS LIST ===

refreshBtn.addEventListener('click', loadDocuments);

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/api/documents`, {
            headers: {
                'x-session-id': sessionId
            }
        });
        const data = await response.json();
        if (response.ok) displayDocuments(data.documents);
        else console.error('Load failed');
    } catch (e) {
        console.error(e);
    }
}

function displayDocuments(documents) {
    if (!documents || documents.length === 0) {
        documentsList.innerHTML = `<div style="padding:24px;text-align:center;color:#444;font-size:12px;">INDEX EMPTY</div>`;
        return;
    }

    documentsList.innerHTML = documents.map(doc => `
        <div class="document-card">
            <div class="document-info">
                <div class="document-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                <div class="document-meta">${doc.total_chunks} CHUNKS · ${doc.file_type.toUpperCase()}</div>
            </div>
            <button class="delete-btn" onclick="deleteDocument('${doc.document_id}', '${escapeHtml(doc.filename).replace(/'/g, "\\'")}')">DELETE</button>
        </div>
    `).join('');
}

async function deleteDocument(id, name) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
        const res = await fetch(`${API_BASE}/api/documents/${id}`, {
            method: 'DELETE',
            headers: {
                'x-session-id': sessionId
            }
        });
        if (res.ok) { showToast('Document deleted', 'success'); loadDocuments(); }
    } catch (e) { showToast('Error deleting', 'error'); }
}


// === HELPERS ===

function showToast(msg, type = 'success') {
    toast.textContent = msg;
    toast.className = `toast toast-${type} show`;
    setTimeout(() => {
        toast.className = 'toast'; // Hide
    }, 3000);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Init
loadDocuments();
console.log('System Ready v4.0 (Multi-Tenant)');
