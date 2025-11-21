document.getElementById('send-btn').addEventListener('click', function() {
  const input = document.getElementById('chat-input');
  const log = document.getElementById('chat-log');
  if(input.value.trim()!==''){
    const userMsg=document.createElement('div');
    userMsg.textContent=`You: ${input.value}`;
    userMsg.style.textAlign='right'; userMsg.style.marginBottom='10px';
    log.appendChild(userMsg);

    const botMsg=document.createElement('div');
    botMsg.textContent='AI: This is a placeholder response.';
    botMsg.style.textAlign='left'; botMsg.style.marginBottom='10px'; botMsg.style.opacity=0;
    log.appendChild(botMsg);

    setTimeout(()=>{ botMsg.style.transition='opacity 0.5s'; botMsg.style.opacity=1; },300);

    input.value='';
    log.scrollTop=log.scrollHeight;
  }
});
