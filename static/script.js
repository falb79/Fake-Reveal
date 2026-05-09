document.addEventListener('DOMContentLoaded', () => {
  console.log("JS is connected successfully");

  // define all necessary elements 
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browseBtn');
  const uploadBoxContainer = document.getElementById('upload-box-container');
  const spinnerBox = document.getElementById('spinner-box');
  const resultBox = document.getElementById('result-box');
  const preview = document.getElementById('preview');
  const resultLabel = document.getElementById('result-label');
  const resultPercentage = document.getElementById('result-percentage');
  const localization = document.getElementById('localization');
  const analysis = document.getElementById('analysis');
  const audioText = document.getElementById('audio-text');
  const visualText = document.getElementById('visual-text');
  const viewAnalysisBtn = document.getElementById('view-analysis');
  const hideAnalysisBtn = document.getElementById('hide-analysis');
  const localizationHint = document.getElementById('localization-hint');
  const uploadAgainBtn = document.getElementById('upload-again');

  // Copy Result elements
  const copyResultBtn = document.getElementById('copy-result');
  const copyMessage = document.getElementById('copy-message');

  let latestResult = {
    prediction: "",
    confidence: "",
    date: ""
  };

  const errorBox = document.getElementById('error-box');
  const errorMessage = document.getElementById('error-message');
  const retryBtn = document.getElementById('retry-btn');
  const formatsText = document.getElementById('formats');
  const sections = document.querySelectorAll(".section");
  const navLinks = document.querySelectorAll(".nav-link");
  const percentText = document.getElementById('upload-percentage');

  ///--------------Nav-bar Section ---------------

  //when more than half of the section is visible
  const options = {
    threshold: 0.6
  };

  //intersection observer is used to detect the visible section when scrolling
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        //remove active class from all links
        navLinks.forEach(link => link.classList.remove("active"));

        //add active class to the current link
        const id = entry.target.getAttribute("id");
        document.querySelector(`a[href="#${id}"]`).classList.add("active");
      }
    });
  }, options);

  //observe each section
  sections.forEach(section => observer.observe(section));

  //mobile menu toggle
  const menuBtn = document.getElementById("menu-btn");
  const navMenu = document.getElementById("nav-menu");

  menuBtn.addEventListener("click", () => {
    navMenu.classList.toggle("active");
  });

  //------------Upload Section ------------

  // valid extensions for video
  const validExt = {
    video: ['mp4', 'mov']
  };

  let currentType = 'video';

  // trigger the file input click 
  browseBtn.addEventListener('click', () => {
    fileInput.click();
  });

  // Browse file upload
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleVideoFile(file);
  });

  // Ctrl + V paste upload
  document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;

    if (!items) {
      showPasteError();
      return;
    }

    for (const item of items) {
      if (item.type.startsWith("video/")) {
        const file = item.getAsFile();

        if (file) {
          handleVideoFile(file);
          return;
        }
      }
    }

    showPasteError();
  });

  // add event listener to the retry button in the error box
  retryBtn.addEventListener('click', () => {
    fileInput.value = '';
    errorBox.style.display = 'none';
    resultBox.style.display = 'none';
    uploadBoxContainer.style.display = 'block';
  });

  // add event listener to upload again button 
  uploadAgainBtn.addEventListener('click', () => {
    fileInput.value = '';
    preview.innerHTML = '';
    resultBox.style.display = 'none';
    errorBox.style.display = 'none';
    spinnerBox.style.display = 'none';
    uploadBoxContainer.style.display = 'block';
    percentText.textContent = '0%';

    if (copyMessage) {
      copyMessage.textContent = '';
    }

    latestResult = {
      prediction: "",
      confidence: "",
      date: ""
    };
  });

  // Main function for Browse and Ctrl + V
 // أضفنا كلمة async هنا
async function handleVideoFile(file) {
    
    // أضفنا كلمة await هنا لانتظار نتيجة التحقق (الحجم، الصيغة، والصوت)
    const isValid = await validateFile(file, currentType);
    if (!isValid) return; 

    // إذا وصل الكود هنا، يعني الملف سليم ومطابق للشروط
    uploadBoxContainer.style.display = 'none';
    spinnerBox.style.display = 'flex';
    resultBox.style.display = 'none';
    errorBox.style.display = 'none';
    preview.innerHTML = '';
    percentText.textContent = '0%';

    if (copyMessage) {
        copyMessage.textContent = '';
    }

    const formData = new FormData();
    formData.append("video", file);

    const url = URL.createObjectURL(file);
    const videoElement = document.createElement('video');
    videoElement.src = url;
    videoElement.controls = true;
    videoElement.style.background = "#000";

    preview.appendChild(videoElement);

    fetchData(formData);
}

  // function to validate the file type
  async function validateFile(file, type) {
    if (!file) return false;

    const ext = file.name.toLowerCase().split('.').pop();

    // check extension first
    if (!validExt[type].includes(ext)) {
      showError(`Unsupported file type. <br> Supported: ${validExt[type].join(', ').toUpperCase()}`);
      return false;
    }

    //  check file size 
    const maxSizeInMB = 13;
    if (file.size > maxSizeInMB * 1024 * 1024) {
      showError(`File is too large. <br> Maximum size allowed is ${maxSizeInMB}MB.`);
      return false;
    }

    //  check for audio track
    if (type === 'video' || ext === 'mp4') { // make sure the type name is correct
      const hasAudio = await checkAudioTrack(file);
      if (!hasAudio) {
        showError("This video has no audio track. <br> Multimodal analysis requires audio.");
        return false;
      }
    }

    return true;
  }

// function to check if the video has an audio track (using multiple properties for better compatibility)
function checkAudioTrack(file) {
  return new Promise((resolve) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const reader = new FileReader();

    reader.onload = function() {
      // محاولة فك تشفير البيانات كصوت
      audioContext.decodeAudioData(reader.result, 
        (buffer) => {
          // إذا نجح فك التشفير، نتحقق هل عدد القنوات أكبر من 0
          const hasAudio = buffer.numberOfChannels > 0;
          audioContext.close();
          resolve(hasAudio);
        }, 
        (error) => {
          // إذا فشل فك التشفير، غالباً لا يوجد مسار صوتي
          audioContext.close();
          resolve(false);
        }
      );
    };

    reader.onerror = () => {
      audioContext.close();
      resolve(false);
    };

    // نقرأ أول جزء من الملف فقط لتسريع العملية
    reader.readAsArrayBuffer(file);
  });
}

  function showError(message) {
    errorMessage.innerHTML = message;
    errorBox.style.display = 'block';
    uploadBoxContainer.style.display = 'none';
    spinnerBox.style.display = 'none';
    resultBox.style.display = 'none';
    fileInput.value = '';
  }

  // Error for Ctrl + V
  function showPasteError() {
    errorMessage.innerHTML = `
    No video found in clipboard.<br>
    <span class="supported-formats">
      Copy a video file first, then press Ctrl + V.
    </span>
  `;

    errorBox.style.display = 'block';
    uploadBoxContainer.style.display = 'none';
    spinnerBox.style.display = 'none';
    resultBox.style.display = 'none';
  }

  // Old function for Sending request to backend (without percentage bar)
  // async function fetchData(formData) {
  //   try {
  //     const response = await fetch('/send_to_colab', {
  //       method: 'POST',
  //       body: formData
  //     });

  //     if (!response.ok) {
  //       throw new Error(`HTTP error! status: ${response.status}`);
  //     }

  //     const data = await response.json();
  //     console.log("Data received:", data);

  //     // IMPORTANT: use correct keys
  //     showResult(data.prediction, data.confidence_score, data.segments);

  //     console.log(
  //       "Similarity:", data.similarity,
  //       "\nWER:", data.wer,
  //       "\nCorrect words:", data.correct_words
  //     );

  //   } catch (error) {
  //     console.error("Error fetching data:", error);
  //     alert("An error occurred while processing the file.");
  //   } finally {
  //     spinnerBox.style.display = 'none';
  //     resultBox.style.display = 'block';
  //     // uploadText.style.display = 'none';
  //   }
  // }

  // New function for sending request to backend with percentage bar
  function fetchData(formData) {
    // promise to handle the upload progress and server response
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      let currentPercent = 0;

      //Incremental progress simulation
      const progressInterval = setInterval(() => {
        //from 0% to 95% until the server responds
        if (currentPercent < 95) {
          currentPercent++;
          percentText.textContent = `${currentPercent}%`;
        }
      }, 400); //  every 400ms (less than second)

      xhr.onreadystatechange = () => {
        if (xhr.readyState === 4) { // 4 means the request is done
          // Stop the progress simulation 
          clearInterval(progressInterval);

          if (xhr.status === 200) {
            // Ensure the progress bar reaches 100% before showing results
            percentText.textContent = `100%`;

            setTimeout(() => {
              spinnerBox.style.display = 'none';
              resultBox.style.display = 'block';
              currentPercent = 0;
              const data = JSON.parse(xhr.responseText);
              console.log(data);
              if (data.prediction === "Fake") {
                localization.style.display = "inline-block";
                // localization video
                const player = document.getElementById('player');
                const source = document.getElementById('videoSource');
                player.style.display = 'none';
                if (data.video_url) {
                  source.src = `http://127.0.0.1:8000${data.video_url}`;
                  player.load();
                  player.style.display = 'inline-block';
                } else {
                  alert("Processing failed.");
                }

              }
              // display the results
              showResult(data.prediction, data.confidence_score, data.segments, data.whisper_text, data.avhubert_text);
              resolve(data);
            }, 1000);
          } else {
            spinnerBox.style.display = 'none';
            errorMessage.innerHTML = `Sorry, an error occurred while processing the file.<br><span class="supported-formats">Please try again.</span>`;
            errorBox.style.display = 'block';
            currentPercent = 0;
            reject(xhr.statusText);
          }
        }
      };
      xhr.open('POST', '/send_to_colab', true);
      xhr.send(formData);
    });
  }

  // function to add the results to the corresponding elements 
  function showResult(label, score, segments, audio_text, visual_text) {
    resultLabel.textContent = label;
    // display the corresponding elements when the result is Fake
    if (label === "Fake") {
      resultLabel.classList.add("result-fake");

      // list of mismatch audio words 
      const mismatch_audio = segments.map(item => item["audio_word"].toLowerCase());

      // process the Whisper text to highlight differences 
      const processedAudioText = audio_text.split(' ').map(word => {
        // remove punctuation for the check
        const cleanWord = word.toLowerCase().replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, "");
        // if the word is one of the mismatch words, highlight it
        if (mismatch_audio.includes(cleanWord)) {
          return `<span class="highlight">${word}</span>`;
        }
        return word;
      }).join(' ');


      // list of mismatch visual words 
      const mismatch_visual = segments.map(item => item["visual_word"].toLowerCase());

      // process the AV-HuBERT text to highlight differences 
      const processedVisualText = visual_text.split(' ').map(word => {
        // remove punctuation for the check
        const cleanWord = word.toLowerCase().replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, "");
        // if the word is one of the mismatch words, highlight it
        if (mismatch_visual.includes(cleanWord)) {
          return `<span class="highlight">${word}</span>`;
        }
        return word;
      }).join(' ');

      audioText.innerHTML = processedAudioText;
      visualText.innerHTML = processedVisualText;
    }
    else {
      resultLabel.classList.add("result-real");
    }

    resultPercentage.textContent = `${score * 100}%`;

    latestResult = {
      prediction: label,
      confidence: `${score * 100}%`,
      date: new Date().toLocaleString()
    };

    if (copyMessage) {
      copyMessage.textContent = "";
    }
  }

  // Copy Result button
  if (copyResultBtn) {
    copyResultBtn.addEventListener('click', async () => {
      const resultText = `
Fake Reveal Detection Result
----------------------------
Prediction: ${latestResult.prediction}
Confidence Score: ${latestResult.confidence}
Date: ${latestResult.date}
      `.trim();

      try {
        await navigator.clipboard.writeText(resultText);

        if (copyMessage) {
          copyMessage.textContent = "Result copied successfully!";
          copyMessage.style.color = "#2ecc71";
        }

      } catch (error) {
        if (copyMessage) {
          copyMessage.textContent = "Could not copy the result. Please try again.";
          copyMessage.style.color = "#e74c3c";
        }
      }
    });
  }

  // event listener to view the analysis results of fake videos
  viewAnalysisBtn.addEventListener('click', () => {
    analysis.style.display = "inline-block";
    hideAnalysisBtn.style.display = "inline-block";
    viewAnalysisBtn.style.display = "none";
  });
  // event listener to hide analysis results of fake videos
  hideAnalysisBtn.addEventListener('click', () => {
    analysis.style.display = "none";
    hideAnalysisBtn.style.display = "none";
    viewAnalysisBtn.style.display = "inline-block";
    resultBox.scrollIntoView();
  });

  // event listener to handle the display of the video localization description
  localizationHint.addEventListener('click', () => {
    const element = document.getElementById("localization-hint-output");
    if (element.style.display === "none") {
      element.style.display = "block";
    }
    else {
      element.style.display = "none";
    }
  });

  // add event listener to upload again button 
  uploadAgainBtn.addEventListener('click', () => {
    fileInput.value = '';
    preview.innerHTML = '';
    resultBox.style.display = 'none';
    uploadBoxContainer.style.display = 'block';
    // uploadText.style.display = 'block';

    if (copyMessage) {
      copyMessage.textContent = '';
    }

    latestResult = {
      prediction: "",
      confidence: "",
      date: ""
    };
  });

  // function to validate the file type based on the current selection
  // function validateFile(file, type) {
  //   if (!file) return false;

  //   const ext = file.name.toLowerCase().split('.').pop();
  //   if (!validExt[type].includes(ext)) {
  //     errorMessage.innerHTML = `Unsupported file type.<br><span class="supported-formats">Supported formats are: ${validExt[type].join(', ').toUpperCase()}</span>`;
  //     errorBox.style.display = 'block';
  //     uploadBoxContainer.style.display = 'none';
  //     // uploadText.style.display = 'none';
  //     fileInput.value = '';
  //     return false;
  //   }
  //   return true;
  // }

  // -------------Detection Layers Section -----------
  const cards = document.querySelectorAll('.layer-card');
  const connectors = document.querySelectorAll('.layer-connector');
  const resultCard = document.getElementById('resultCard');
  let lastActive = -1;

  function updateCards() {
    const wh = window.innerHeight;
    let activeIndex = -1;

    cards.forEach((card, i) => {
      const rect = card.getBoundingClientRect();
      if (rect.top + rect.height / 2 < wh * 0.58) activeIndex = i;
    });

    if (activeIndex === lastActive) return;
    lastActive = activeIndex;

    cards.forEach((card, i) => {
      card.classList.remove('active', 'past');
      if (i === activeIndex) card.classList.add('active');
      else if (i < activeIndex) card.classList.add('past');
    });

    connectors.forEach((conn, i) => {
      conn.classList.toggle('visible', i < activeIndex);
    });

    resultCard.classList.toggle('show', activeIndex === cards.length - 1);
  }

  window.addEventListener('scroll', updateCards, { passive: true });
  updateCards();

  //-------------- Game Section (updated )------------

  const media = [
    {
      src: "../static/assets/f1.mp4",
      type: "Fake"
    },
    {
      src: "../static/assets/r1.mp4",
      type: "Real"
    },
    {
      src: "../static/assets/f2.mp4",
      type: "Fake"
    },
    {
      src: "../static/assets/r2.mp4",
      type: "Real"
    },
    {
      src: "../static/assets/f3.mp4",
      type: "Fake"
    },
    {
      src: "../static/assets/r3.mp4",
      type: "Real"
    },
    {
      src: "../static/assets/f4.mp4",
      type: "Fake"
    },
    {
      src: "../static/assets/r4.mp4",
      type: "Real"
    },
    {
      src: "../static/assets/f5.mp4",
      type: "Fake"
    },
    {
      src: "../static/assets/r5.mp4",
      type: "Real"
    }
  ];

  // Shuffle videos
  function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  shuffle(media);

  let currentIndex = 0;
  let score = 0;

  const demoWrapper = document.getElementById("demo-wrapper");
  const demoDots = document.getElementById("demo-dots");
  const feedback = document.getElementById("demo-feedback");
  const realBtn = document.getElementById("real-btn");
  const fakeBtn = document.getElementById("fake-btn");
  const demoQuestion = document.querySelector(".demo-question");
  const liveScore = document.getElementById("live-score");

  // Show current video
  function showMedia(index) {
    const item = media[index];

    demoWrapper.innerHTML = `
    <div class="demo-card big-card">
      <video
        controls
        style="width:100%; height:100%; border-radius:18px; object-fit:contain; background:#000;">
        <source src="${item.src}" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>
  `;

    updateDots(index);
    feedback.textContent = "";
    demoQuestion.textContent = `Video ${index + 1} out of ${media.length}`;

    if (liveScore) {
      liveScore.textContent = `Score: ${score} / ${media.length}`;
    }

    realBtn.style.display = "inline-block";
    fakeBtn.style.display = "inline-block";
    realBtn.disabled = false;
    fakeBtn.disabled = false;
  }

  // Update dots
  function updateDots(index) {
    demoDots.innerHTML = media
      .map((_, i) => `<span class="dot ${i === index ? "active" : ""}"></span>`)
      .join("");
  }

  // Check user's answer
  function checkAnswer(answer) {
    const clickedBtn = answer === "Real" ? realBtn : fakeBtn;

    if (answer === media[currentIndex].type) {
      score++;
      feedback.textContent = "You got it right!";
      clickedBtn.style.backgroundColor = "#2ecc71";
      clickedBtn.style.color = "#fff";
    } else {
      feedback.textContent = "Oops! Wrong!";
      clickedBtn.style.backgroundColor = "#e74c3c";
      clickedBtn.style.color = "#fff";
    }

    if (liveScore) {
      liveScore.textContent = `Score: ${score} / ${media.length}`;
    }

    realBtn.disabled = true;
    fakeBtn.disabled = true;

    setTimeout(() => {
      clickedBtn.style.backgroundColor = "";
      clickedBtn.style.color = "";

      currentIndex++;

      if (currentIndex >= media.length) {
        showFinalScore();
      } else {
        showMedia(currentIndex);
      }
    }, 1500);
  }

   //Show final score
  function showFinalScore() {
    demoWrapper.innerHTML = `
    <div class="demo-card big-card final-score-card">
      <div>
        <h2>Game Finished!</h2>
        <p>Your Score:</p>
        <h1>${score} / ${media.length}</h1>

        <p class="system-score">
          Fake Reveal system achieved 10/10 score by correctly classifying all videos .
        </p>

        <button id="play-again-btn">Play Again</button>
      </div>
    </div>
  `;

    if (liveScore) {
      liveScore.textContent = "";
    }

    demoDots.innerHTML = "";
    demoQuestion.textContent = "";
    feedback.textContent = "";

    realBtn.style.display = "none";
    fakeBtn.style.display = "none";

    document.getElementById("play-again-btn").addEventListener("click", restartGame);
  }

  //Restart game
  function restartGame() {
    currentIndex = 0;
    score = 0;
    shuffle(media);

    if (liveScore) {
      liveScore.textContent = `Score: 0 / ${media.length}`;
    }

    showMedia(currentIndex);
  }


  //Buttons
  realBtn.addEventListener("click", () => checkAnswer("Real"));
  fakeBtn.addEventListener("click", () => checkAnswer("Fake"));

  //First run
  showMedia(currentIndex);

  ///--------------- FAQ Section ------------
  const faqItems = document.querySelectorAll('.faq-item');

  faqItems.forEach(item => {
    const btn = item.querySelector('.faq-question');
    const answer = item.querySelector('.faq-answer');

    btn.addEventListener('click', () => {
      const isOpen = answer.style.maxHeight && answer.style.maxHeight !== "0px";

      document.querySelectorAll('.faq-answer').forEach(ans => {
        ans.style.maxHeight = null;
      });

      document.querySelectorAll('.faq-question').forEach(question => {
        question.classList.remove('active');
      });

      if (!isOpen) {
        answer.style.maxHeight = answer.scrollHeight + "px";
        btn.classList.add('active');
      }
    });
  });


});


