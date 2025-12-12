// define all necessary elements 
const imageBtn = document.getElementById('image-btn');
const videoBtn = document.getElementById('video-btn');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browseBtn');
const uploadBoxContainer = document.getElementById('upload-box-container');
const spinnerBox = document.getElementById('spinner-box');
const resultBox = document.getElementById('result-box');
const preview = document.getElementById('preview');
const resultLabel = document.getElementById('result-label');
const resultPercentage = document.getElementById('result-percentage');
const uploadAgainBtn = document.getElementById('upload-again');
const errorBox = document.getElementById('error-box');
const errorMessage = document.getElementById('error-message');
const retryBtn = document.getElementById('retry-btn');
const uploadText = document.getElementById('upload-text');
const formatsText = document.getElementById('formats');

// valid extensions for image and video
const validExt = {
  image: ['png', 'jpg', 'jpeg'],
  video: ['mp4', 'mov']
};

// default the current user selection to image
let currentType = 'video';

// trigger the file input click 
browseBtn.addEventListener('click', () => {
  fileInput.click();
});

// add event listener to the image button to change the current selection
// and update the accepted file types
imageBtn.addEventListener('click', () => {
  currentType = 'image';
  imageBtn.classList.add('active');
  videoBtn.classList.remove('active');
  fileInput.accept = '.png, .jpg, .jpeg';
  formatsText.textContent = "Supported Image Formats are: PNG, JPG, JPEG";
});

// add event listener to the video button to change the current selection
// and update the accepted file types
videoBtn.addEventListener('click', () => {
  currentType = 'video';
  videoBtn.classList.add('active');
  imageBtn.classList.remove('active');
  fileInput.accept = '.mp4, .mov';
  formatsText.textContent = "Supported video Formats are: MP4, MOV";
});

// add event listener to the retry button in the error box
retryBtn.addEventListener('click', () => {
  fileInput.value = '';
  errorBox.style.display = 'none';
  uploadBoxContainer.style.display = 'block';
  uploadText.style.display = 'block';
});

// add event listener to the file input element 
fileInput.addEventListener('change', (e) => {
  // get the uploaded file
  const file = e.target.files[0];
  //call validate function to check file existence and type
  if (!validateFile(file, currentType)) return;


  // Hide upload box and show spinner
  uploadBoxContainer.style.display = 'none';
  spinnerBox.style.display = 'flex';
  resultBox.style.display = 'none';
  preview.innerHTML = '';

  // create formdata objects for image and video to be used to fetch the data
  const imageFormData = new FormData();
  const videoFormData = new FormData();
  const url = URL.createObjectURL(file);
  let element;

  // create image object if it's the current selection
  if (currentType === 'image') {
    element = document.createElement('img');
    element.src = url;
    // add it to the formdata object
    imageFormData.append("image", file);
    // fetch the data when the image is loaded
    element.onload = () => {
      fetchData(element, "predict_image", imageFormData);
    };
  }
  // create video object if it's the current selection
  else {
    element = document.createElement('video');
    element.src = url;
    element.controls = true;
    // add it to the formdata object
    videoFormData.append("video", file);
    // fetch the data when the video is loaded
    element.onloadeddata = () => {
      fetchData(element, "predict_video", videoFormData);
    };
  }
});

// function to fetch the data from the model's endpoints 
async function fetchData(element, endpoint, formData) {
  // define the url to the correct route
  const url = `http://localhost:5000/${endpoint}`;
  try {
    // send the request and wait for the response
    const response = await fetch(url, {
      method: "POST",
      body: formData
    });
    // handle http errors
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json(); // parse the JSON response
    // call the showResult function to display the results 
    showResult(data["label"], data["score"]);
    console.log("Label:",data["label"],"\nScore:" ,data["score"],"\nLip_reading_text:",data["lip_reading_text"],".\n Speech_text:",data["speech_text"]);
  } catch (error) {
    console.error("Error fetching data:", error);
    alert("An error occurred while processing the file. Please try again.");
  }
  finally {
    // hide the spinner and display the uploaded file
    spinnerBox.style.display = 'none';
    preview.appendChild(element);
    resultBox.style.display = 'block';
    uploadText.style.display = 'none';
  }
}

// function to add the results to the corresponding elements 
function showResult(label, score) {
  resultLabel.textContent = label;
  resultPercentage.textContent = score;
}

// add event listener to upload again button 
uploadAgainBtn.addEventListener('click', () => {
  fileInput.value = '';
  preview.innerHTML = '';
  resultBox.style.display = 'none';
  uploadBoxContainer.style.display = 'block';
  uploadText.style.display = 'block';
});

// function to validate the file type based on the current selection
function validateFile(file, type) {
  if (!file) return false;

  const ext = file.name.toLowerCase().split('.').pop();
  if (!validExt[type].includes(ext)) {
    errorMessage.innerHTML = `Unsupported file type.<br><span class="supported-formats">Supported formats are: ${validExt[type].join(', ').toUpperCase()}</span>`;
    errorBox.style.display = 'block';
    uploadBoxContainer.style.display = 'none';
    uploadText.style.display = 'none';
    fileInput.value = '';
    return false;
  }
  return true;
}


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

// randomly display the images in the game section
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

// display the images in the game section
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

// check user's answers 
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

    if (!isOpen) {
      answer.style.maxHeight = answer.scrollHeight + "px";
    }
  });
});

///Nav-bar Section -->
const sections = document.querySelectorAll(".section");
const navLinks = document.querySelectorAll(".nav-link");

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

