class Chatbox {
    constructor() {
        this.args = {
            openButton: document.getElementById('chatButton'),
            chatBox: document.getElementById('chatContainer'),
            closeButton: document.getElementById('chatClose'),
            sendInput: document.getElementById('messageInput'),
            chatMessages: document.getElementById('chatMessages'),
        };

        this.state = false;
        this.messages = [];
    }

    display() {
        const { openButton, chatBox, closeButton, sendInput } = this.args;

        openButton.addEventListener('click', () => this.toggleChat());
        closeButton.addEventListener('click', () => this.closeChat());

        sendInput.addEventListener("keyup", (e) => {
            if (e.key === "Enter") {
                this.sendMessage();
            }
        });
    }

    toggleChat() {
        const chatBox = this.args.chatBox;
        this.state = !this.state;

        if (this.state) {
            chatBox.classList.add('show-chat');
            chatBox.style.display = 'flex';
        } else {
            this.closeChat();
        }
    }

    closeChat() {
        const chatBox = this.args.chatBox;
        chatBox.classList.remove('show-chat');
        chatBox.classList.add('hide-chat');
        setTimeout(() => {
            chatBox.style.display = 'none';
            chatBox.classList.remove('hide-chat');
        }, 300);
        this.state = false;
    }
    
    sendMessage() {
        const { sendInput, chatMessages } = this.args;
        const message = sendInput.value.trim();

        if (message === '') return;

        this.addMessage(message, 'user');
        sendInput.value = '';
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'bot-message');
        loadingDiv.innerHTML = '<div class="loading-dots"><span>.</span><span>.</span><span>.</span></div>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // POST to Flask backend
        fetch('http://127.0.0.1:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: message })
        })
        .then(res => {
            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }
            return res.json();
        })
        .then(data => {
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);
            
            if (data.type === "error") {
                this.addMessage(`❌ Error: ${data.response}`, 'bot');
                return;
            }

            let reply = "";
            
            // Check if this is a blood-related response
            if (data.type === "symptom" && data.response.diagnosis.toLowerCase().includes('blood')) {
                reply = this.formatBloodResponse(data);
            } 
            // Check if this is a medicine-related response
            else if (data.type === "symptom" && 
                    (data.response.diagnosis.toLowerCase().includes('side effects') || 
                     data.response.diagnosis.toLowerCase().includes('medicine'))) {
                reply = this.formatMedicineResponse(data);
            }
            // Default to symptom response
            else {
                reply = this.formatSymptomResponse(data);
            }
            
            this.addMessage(reply, 'bot');
        })
        .catch(err => {
            console.error('Error:', err);
            chatMessages.removeChild(loadingDiv);
            this.addMessage("Error: Could not get a response from the server.", 'bot');
        });
    }

    formatSymptomResponse(data) {
        const response = data.response || {};
        const diagnosis = response.diagnosis || 'No specific diagnosis available';
        const adviceItems = response.advice || ['No specific advice available'];
        const testItems = response.tests || [];

        let html = `
            <div class="response symptom">
                <div class="diagnosis">
                    <h3>🧠 Suggested Diagnosis</h3>
                    <p>${diagnosis}</p>
                </div>
                
                <div class="advice">
                    <h3>💡 Health Advice</h3>
                    <ul>
                        ${adviceItems.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
        `;

        // Add tests section only if tests exist
        if (testItems.length > 0) {
            html += `
                <div class="tests">
                    <h3>🧪 Recommended Tests</h3>
                    <ul>
                        ${testItems.map(test => `<li>${test}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        html += `</div>`;
        return html;
    }

    formatMedicineResponse(data) {
        // Extract medicine name from diagnosis
        const medNameMatch = data.response.diagnosis.match(/Side Effects of (.+)|Information about (.+)/i);
        const medName = (medNameMatch && (medNameMatch[1] || medNameMatch[2])) || 'this medicine';
        
        // Extract buy link if available
        const buyLinkMatch = data.text.match(/🔗 Best Buy: \[(.+?)\]\((.+?)\)/);
        const buyLink = buyLinkMatch ? { text: buyLinkMatch[1], url: buyLinkMatch[2] } : null;

        // Separate side effects from general advice
        const sideEffects = data.response.advice.filter(a => a.toLowerCase().includes('side effect'));
        const generalAdvice = data.response.advice.filter(a => !a.toLowerCase().includes('side effect'));

        return `
            <div class="response medicine">
                <div class="medicine-header">
                    <h3>💊 ${data.response.diagnosis}</h3>
                    ${buyLink ? `<a href="${buyLink.url}" target="_blank" class="buy-link">Buy ${buyLink.text}</a>` : ''}
                </div>
                
                <div class="medicine-details">
                    ${sideEffects.length > 0 ? `
                    <div class="side-effects">
                        <h4>⚠️ Side Effects</h4>
                        <ul>
                            ${sideEffects.map(effect => 
                                `<li>${effect.replace(/side effects?/i, '').trim()}</li>`
                            ).join('')}
                        </ul>
                    </div>
                    ` : ''}
                    
                    ${generalAdvice.length > 0 ? `
                    <div class="medicine-advice">
                        <h4>📝 Usage Advice</h4>
                        <ul>
                            ${generalAdvice.map(advice => `<li>${advice}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>
                
                ${data.response.tests && data.response.tests.length > 0 ? `
                    <div class="recommended-tests">
                        <h4>🧪 Recommended Tests</h4>
                        <ul>
                            ${data.response.tests.map(test => `<li>${test}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }

    formatBloodResponse(data) {
        const bloodBanks = this.extractBloodBankInfo(data.text);
        
        return `
            <div class="response blood">
                <h3>🩸 ${data.response.diagnosis}</h3>
                
                ${data.response.advice && data.response.advice.length > 0 ? `
                <div class="blood-advice">
                    <h4>💡 Health Advice</h4>
                    <ul>
                        ${data.response.advice.map(advice => `<li>${advice}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
                
                <div class="table-responsive">
                    <table class="blood-bank-table">
                        <thead>
                            <tr>
                                <th>S.No.</th>
                                <th>Name</th>
                                <th>Address</th>
                                <th>Phone</th>
                                <th>Email</th>
                                <th>Category</th>
                                <th>Distance</th>
                                <th>Type</th>
                                <th>Stock</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${bloodBanks.map((bank, index) => `
                                <tr>
                                    <td>${index + 1}</td>
                                    <td>${bank.name || 'N/A'}</td>
                                    <td>${bank.address || 'N/A'}</td>
                                    <td>${bank.phone || 'N/A'}</td>
                                    <td>${bank.email || 'N/A'}</td>
                                    <td>${bank.category || 'Private'}</td>
                                    <td>${bank.distance || '-'}</td>
                                    <td>${bank.type || 'Camps'}</td>
                                    <td>${bank.stock || 'Available'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    extractBloodBankInfo(text) {
        const bloodBanks = [];
        const regex = /Name of Unit\/Center: (.+?)\nAddress: (.+?)\nContact Number: (.+?)\nEmail ID: (.+?)(?:\n|$)/g;
        let match;
        
        while ((match = regex.exec(text)) !== null) {
            bloodBanks.push({
                name: match[1].trim(),
                address: match[2].trim(),
                phone: match[3].trim(),
                email: match[4].trim(),
                category: 'Government',
                distance: '-',
                type: 'Camps',
                stock: 'Available'
            });
        }
        
        return bloodBanks;
    }

    addMessage(content, sender) {
        const { chatMessages } = this.args;
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);

        // Check if content is HTML or plain text
        if (typeof content === 'string' && (content.startsWith('<') || content.includes('<div'))) {
            messageDiv.innerHTML = content;
        } else {
            messageDiv.textContent = content;
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Initialize chatbox when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const chatbox = new Chatbox();
    chatbox.display();
});
   

    

// Initialize chatbox when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const chatbox = new Chatbox();
    chatbox.display();
});
// Typing Animation for Bot
function simulateTyping(response, delay = 40) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot-message');
    chatMessages.appendChild(messageDiv);

    let i = 0;
    const interval = setInterval(() => {
        messageDiv.textContent += response[i];
        i++;
        if (i >= response.length) {
            clearInterval(interval);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }, delay);
}
const chatbox = new Chatbox();
chatbox.display();


// Start recording
function startRecording() {
    if (recognition) {
        recognition.start();
        isRecording = true;
        voiceButton.classList.add('recording');
        voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
        if (!isVoiceMode) {
            addMessage('Listening...', 'bot');
        }
    }
}

// Stop recording
function stopRecording() {
    if (recognition) {
        recognition.stop();
        isRecording = false;
        voiceButton.classList.remove('recording');
        voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
    }
}

// Voice button click handler
voiceButton.addEventListener('click', () => {
    if (isRecording) {
        stopAudioRecording();
    } else {
        startAudioRecording();
    }
});

// Typing Animation for Bot
function simulateTyping(response, delay = 40) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot-message');
    chatMessages.appendChild(messageDiv);

    let i = 0;
    const interval = setInterval(() => {
        messageDiv.textContent += response[i];
        i++;
        if (i >= response.length) {
            clearInterval(interval);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }, delay);
}

function loadChatHistory() {
    fetch('/chatlog')
        .then(response => response.json())
        .then(lines => {
            lines.forEach(line => {
                const isUser = line.trim().startsWith("User:");
                const cleanLine = line.replace(/^User:\s*|^Bot:\s*/, '').trim();
                if (cleanLine) {
                    addMessage(cleanLine, isUser ? 'user' : 'bot');
                }
            });
            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(err => {
            console.error("Failed to load chat history:", err);
        });
}

document.addEventListener('DOMContentLoaded', () => {
    setupSpeechRecognition();
    loadChatHistory();
});

function simulateVoiceBotResponse(text) {
    voiceResponse.textContent = '';
    let i = 0;
    const interval = setInterval(() => {
        voiceResponse.textContent += text[i];
        i++;
        if (i >= text.length) {
            clearInterval(interval);
        }
    }, 40);
}

// pdf
function uploadPDF(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('pdf', file);

    fetch('/upload-pdf', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        alert(data.message || 'PDF uploaded and processed.');
    })
    .catch(err => {
        console.error(err);
        alert('Error uploading PDF.');
    });
}

// real voice capture 
let mediaRecorder;
let audioChunks = [];

function startAudioRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        isRecording = true;
        voiceButton.classList.add('recording');
        voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
        voiceModeContainer.style.display = 'flex';
        document.querySelector('.listening-text').innerText = "Listening...";

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');

            fetch('/chat', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                simulateVoiceBotResponse(data.response);
                voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
                isRecording = false;
                voiceButton.classList.remove('recording');
            })
            .catch(err => {
                console.error('Voice request failed:', err);
                simulateVoiceBotResponse("There was an error. Please try again.");
                isRecording = false;
                voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
                voiceButton.classList.remove('recording');
            });
        };

        // Automatically stop recording after 5 seconds
        setTimeout(() => {
            if (mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        }, 5000);
    }).catch(err => {
        console.error('Microphone access denied:', err);
    });
}

function stopAudioRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}
