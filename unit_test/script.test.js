// ---------- validateFile Function ----------

function validateFile(file, type) {
  const validExt = {
    video: ['mp4', 'mov']
  };

  if (!file) return false;

  const ext = file.name.toLowerCase().split('.').pop();

  return validExt[type].includes(ext);
}

// ---------- validateFile Test Cases ----------


// Test Case 1: valid mp4 file
test('validateFile accepts mp4 file', () => {
  const file = { name: 'video.mp4' };

  expect(validateFile(file, 'video')).toBe(true);
});

// Test Case 2: valid mov file
test('validateFile accepts mov file', () => {
  const file = { name: 'movie.mov' };

  expect(validateFile(file, 'video')).toBe(true);
});

// Test Case 3: invalid exe file
test('validateFile rejects exe file', () => {
  const file = { name: 'virus.exe' };

  expect(validateFile(file, 'video')).toBe(false);
});

// Test Case 4: empty file
test('validateFile rejects empty file', () => {
  expect(validateFile(null, 'video')).toBe(false);
});

// ---------- fetchData Function ----------

// fake version to simulate backend response
function fakeFetchData(responseType) {

  if (responseType === "success") {
    return Promise.resolve({
      prediction: "Fake",
      confidence_score: 0.95
    });
  }

  if (responseType === "error") {
    return Promise.reject("Server Error");
  }
}

// ---------- fetchData Test Cases ----------

// Test Case 1: successful response
test('fetchData returns fake prediction successfully', async () => {

  const data = await fakeFetchData("success");

  expect(data.prediction).toBe("Fake");
  expect(data.confidence_score).toBe(0.95);
});

// Test Case 2: failed response
test('fetchData handles server error', async () => {

  await expect(fakeFetchData("error"))
    .rejects
    .toBe("Server Error");
});

// ---------- showResult Function ----------

function showResult(label, score) {
  return `Result: ${label}, Confidence: ${score}`;
}

// ---------- showResult Test Cases ----------

// Test Case 1: Fake result
test('showResult displays fake result correctly', () => {

  const result = showResult("Fake", 95);

  expect(result).toBe("Result: Fake, Confidence: 95");
});

// Test Case 2: Real result
test('showResult displays real result correctly', () => {

  const result = showResult("Real", 88);

  expect(result).toBe("Result: Real, Confidence: 88");
});