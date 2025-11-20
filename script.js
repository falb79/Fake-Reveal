// ===============================
// ====== ADD YOUR API KEY ========
// ===============================
// Put your OpenAI API key here
const OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"; // <-- Replace this with your key
// ===============================

// Upload functionality
document.getElementById('fileInput').addEventListener('change', function() {
  const file = this.files[0];
  const loading = document.getElementById('loading');
  const resultBox = document.getElementById('result-box');
  const resultText = document.getElementById('result-text');
  const progressContainer = document.getElementById('progress-container');
  const progressBar = document.getElementById('progress-bar');
  const uploadedDisplay = document.getElementById('uploaded-display');

  loading.style.display='block';
  progressContainer.style.display='block';
  progressBar.style.width='0%';
  resultBox.style.display='none';
  uploadedDisplay.innerHTML='';

  const url = URL.createObjectURL(file);
  if(file.type.startsWith('image/')){
    const img = document.createElement('img'); img.src=url; uploadedDisplay.appendChild(img);
  } else if(file.type.startsWith('video/')){
    const vid = document.createElement('video'); vid.src=url; vid.controls=true; uploadedDisplay.appendChild(vid);
  }

  let progress=0;
  const interval=setInterval(()=>{
    progress+=Math.floor(Math.random()*10)+5;
    if(progress>100) progress=100;
    progressBar.style.width=progress+'%';
    if(progress>=100){
      clearInterval(interval);
      loading.style.display='none';
      resultBox.style.display='block';
      const fakeChance=Math.floor(Math.random()*100);
      resultText.innerHTML=`Fake Probability: ${fakeChance}%`;
    }
  },300);
});

// Chatbot toggle with slide animation
const toggleBtn = document.getElementById('chatbot-toggle');
const chatWindow = document.getElementById('chatbot-window');

toggleBtn.addEventListener('click', () => {
  chatWindow.classList.toggle('show');
});

// Chatbot send using OpenAI API
document.getElementById('chatbot-send').addEventListener('click', async () => {
  const input = document.getElementById('chatbot-input');
  const log = document.getElementById('chatbot-log');
  if(input.value.trim() !== ''){
    const userMsg = document.createElement('div');
    userMsg.textContent = `You: ${input.value}`;
    userMsg.style.textAlign = 'right';
    userMsg.style.marginBottom='10px';
    log.appendChild(userMsg);

    const botMsg = document.createElement('div');
    botMsg.textContent = 'AI: Processing...';
    botMsg.style.textAlign='left';
    botMsg.style.marginBottom='10px';
    botMsg.style.opacity=0;
    log.appendChild(botMsg);

    try {
      // Call OpenAI API directly
      const response = await fetch("https://api.ai-studio.cloud/xxxxxx/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${gen-lang-client-0388428018}`
        },
        body: JSON.stringify({
          model: "gpt-3.5-turbo",
          messages: [{ role: "user", content: input.value }]
        })
      });
      const data = await response.json();
      botMsg.style.opacity = 1;
      botMsg.textContent = "AI: " + data.choices[0].message.content;
    } catch(err) {
      botMsg.textContent = "AI: Error connecting to API.";
      console.error(err);
    }

    input.value = '';
    log.scrollTop = log.scrollHeight;
  }
});

// Auto-moving carousel
const carousel = document.getElementById('carousel');
let scrollAmount = 0;
setInterval(()=>{
  scrollAmount += 220;
  if(scrollAmount > carousel.scrollWidth - carousel.clientWidth) scrollAmount=0;
  carousel.scrollTo({left:scrollAmount, behavior:'smooth'});
},3000);
