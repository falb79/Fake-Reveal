// upload
const imageBtn = document.getElementById('image-btn');
const videoBtn = document.getElementById('video-btn');
const fileInput = document.getElementById('file-input');
const uploadBoxContainer = document.getElementById('upload-box-container');
const spinnerBox = document.getElementById('spinner-box');
const resultBox = document.getElementById('result-box');
const preview = document.getElementById('preview');
const resultLabel = document.getElementById('result-label');
const resultPercentage = document.getElementById('result-percentage');
const uploadAgainBtn = document.getElementById('upload-again');

let currentType = 'image';

imageBtn.addEventListener('click', () => {
  currentType = 'image';
  imageBtn.classList.add('active');
  videoBtn.classList.remove('active');
  fileInput.accept = 'image/*';
});

videoBtn.addEventListener('click', () => {
  currentType = 'video';
  videoBtn.classList.add('active');
  imageBtn.classList.remove('active');
  fileInput.accept = 'video/*';
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  // Hide upload box and show spinner
  uploadBoxContainer.style.display = 'none';
  spinnerBox.style.display = 'flex';
  resultBox.style.display = 'none';
  preview.innerHTML = '';

  const imageFormData = new FormData();
  const videoFormData = new FormData();
  const url = URL.createObjectURL(file);
  let element;

  if (currentType === 'image') {
    element = document.createElement('img');
    element.src = url;
    imageFormData.append("image", file);
    element.onload = () => {
      fetchData(element, "predict_image", imageFormData);
    };
  } else {
    element = document.createElement('video');
    element.src = url;
    element.controls = true;
    videoFormData.append("video", file);
    element.onloadeddata = () => {
      fetchData(element, "predict_video", videoFormData);
    };
  }
});


async function fetchData(element, endpoint, formData) {
  url = `http://localhost:5000/${endpoint}`; 
  try {
    const response = await fetch(url, {
      method: "POST",
      body: formData 
    }); 
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json(); // Parse the JSON response
    showResult(data["label"], data["score"]);
  } catch (error) {
    console.error("Error fetching data:", error);
  }
  finally{
    spinnerBox.style.display = 'none';
    preview.appendChild(element);
    resultBox.style.display = 'block';
  }
}

function showResult(label, score) {
  resultLabel.textContent = label;
  resultPercentage.textContent = score;
}

// function showResult(element) {
//   spinnerBox.style.display = 'none';
//   preview.appendChild(element);
//   resultBox.style.display = 'block';

//   const outcomes = ['Fake', 'Real'];
//   const randomOutcome = outcomes[Math.floor(Math.random() * outcomes.length)];
//   resultLabel.textContent = randomOutcome;

//   const percent = currentType === 'image' ? Math.floor(Math.random() * 41 + 60) : '-';
//   resultPercentage.textContent = percent + (percent !== '-' ? '%' : '');
// }

uploadAgainBtn.addEventListener('click', () => {
  fileInput.value = '';
  preview.innerHTML = '';
  resultBox.style.display = 'none';
  uploadBoxContainer.style.display = 'block';
});


// Hero Scroll
document.getElementById("get-started").addEventListener("click", function () {
  document.getElementById("upload-demo").scrollIntoView({ behavior: "smooth" });
});

// ================= Game Section (updated ) =================
const images = [
     { src: "https://i.ibb.co/XxR3tptM/real-10005.jpg", type: "Real" },
    { src: "https://i.ibb.co/JF7K8PYh/real-1015.jpg", type: "Real" },
    { src: "https://i.ibb.co/7NJNqj5c/real-1000.jpg", type: "Real" },
    { src: "https://i.ibb.co/20t8nzKX/real-1016.jpg", type: "Real" },
    { src: "https://i.ibb.co/cKKj822S/real-1.jpg", type: "Real" },

     { src: "https://i.ibb.co/KjLWPsYr/fake-12.jpg", type: "Fake" },
    { src: "https://i.ibb.co/csr5ZM7/fake-1009.jpg", type: "Fake" },
    { src: "https://i.ibb.co/tMmkY1q6/fake-1.jpg", type: "Fake" },
    { src: "https://i.ibb.co/1JjRNXv2/fake-1014.jpg", type: "Fake" },
    { src: "https://i.ibb.co/vCSJ9T1t/fake-1006.jpg", type: "Fake" }
];
 
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

shuffle(images);  

let currentIndex = 0;

const demoWrapper = document.getElementById('demo-wrapper');
const demoDots = document.getElementById('demo-dots');
const feedback = document.getElementById('demo-feedback');
const realBtn = document.getElementById('real-btn');
const fakeBtn = document.getElementById('fake-btn');

function showImage(index) {
    demoWrapper.innerHTML = `
        <div class="demo-card big-card">
            <img src="${images[index].src}" 
            style="width:100%; height:100%; border-radius:18px; object-fit:cover;">
        </div>`;
    updateDots(index);
    feedback.textContent = "";
}

function updateDots(index) {
    demoDots.innerHTML = images.map((img, i) =>
        `<span class="dot ${i === index ? 'active' : ''}"></span>`
    ).join('');
}

function checkAnswer(answer) {

    const clickedBtn = answer === "Real" ? realBtn : fakeBtn;

    if (answer === images[currentIndex].type) {
      feedback.textContent = "You got it right!";
      clickedBtn.style.backgroundColor = "#2ecc71";
      clickedBtn.style.color = "#ffffffff";
    } else {
      feedback.textContent = "Oops! Wrong!";
      clickedBtn.style.backgroundColor = "#e74c3c"
      clickedBtn.style.color = "#ffffffff";
    }
    setTimeout(() => {
      clickedBtn.style.backgroundColor = "";
      clickedBtn.style.color = "";
      currentIndex = (currentIndex + 1) % images.length;
      showImage(currentIndex);
    }, 1500);
}

realBtn.addEventListener('click', () => checkAnswer("Real"));
fakeBtn.addEventListener('click', () => checkAnswer("Fake"));

 showImage(currentIndex);
  
///FAQ Section -->
const faqItems = document.querySelectorAll('.faq-item');

faqItems.forEach(item => {
  const btn = item.querySelector('.faq-question');
  const answer = item.querySelector('.faq-answer');

  btn.addEventListener('click', () => {
    const isOpen = answer.style.maxHeight && answer.style.maxHeight !== "0px";

    // close all answers
    document.querySelectorAll('.faq-answer').forEach(ans => ans.style.maxHeight = null);

    if(!isOpen){
      answer.style.maxHeight = answer.scrollHeight + "px";
    }
  });
});

