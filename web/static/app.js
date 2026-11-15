// The web version's behaviour. The browser runs this file. It asks web_app.py for things with
// fetch(), and keeps the page up to date:
//   - it asks for the status (/api/status) a few times a second, and shows it;
//   - while the camera is on, it asks for the newest picture (/api/frame.jpg) again and again;
//   - the buttons and forms send requests that change something, like starting the camera.

"use strict";

const byId = (id) => document.getElementById(id);
let status = null; // the last status the server sent

// A refusal for the camera's message line, like "No one is registered yet". It stays for a few seconds;
// otherwise the next status, a moment later, would replace it with the camera's own message.
let refusal = null;

// ---------------------------------------------------------------- talking to the server

// Every request carries the "X-Face-App" header. The server only accepts changes from requests that
// have it (see before_request in web_app.py), and other websites can't add it.
async function ask(path, { json, form } = {}) {
  const options = { headers: { "X-Face-App": "1" } };
  if (json !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(json);
  } else if (form !== undefined) {
    options.method = "POST";
    options.body = form; // a FormData: the browser sends the files in it
  }
  const response = await fetch(path, options);
  const answer = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(answer.error || `Something went wrong (the server answered ${response.status}).`);
  }
  return answer;
}

// Send a change, then show the new status. A refusal (like a name that isn't allowed) is shown in
// the message line given by `messageId`.
async function act(path, json = {}, messageId = "message") {
  try {
    const answer = await ask(path, { json });
    refusal = null;
    show(answer);
  } catch (problem) {
    if (messageId === "message") {
      refusal = { text: problem.message, until: Date.now() + 8000 };
    }
    say(messageId, problem.message, true);
  }
}

function say(id, text, isProblem = false) {
  const line = byId(id);
  line.textContent = text;
  line.classList.toggle("problem", isProblem);
}

// The same, for a message line that takes no room while it has nothing to say.
function sayOrHide(id, text, isProblem = false) {
  say(id, text, isProblem);
  byId(id).hidden = !text;
}

// Ask for the status again and again: quickly while the camera is on, slowly while it is off.
async function keepStatusUpToDate() {
  try {
    show(await ask("/api/status"));
  } catch (problem) {
    byId("error").textContent = "The program isn't answering. Is it still running in its terminal window?";
    byId("error").hidden = false;
  }
  setTimeout(keepStatusUpToDate, status && status.mode ? 300 : 1000);
}

// ---------------------------------------------------------------- showing the status

function show(newStatus) {
  status = newStatus;
  byId("today").textContent = dateInWords(status.today);
  showCameraList();
  showCamera();
  showPresentToday();
  showPeople();
}

// A date ("2026-11-15") in words, in the computer's language: "Sunday, 15 November", or "Sun, 15 Nov".
function dateInWords(date, short = false) {
  const parts = short
    ? { weekday: "short", day: "numeric", month: "short" }
    : { weekday: "long", day: "numeric", month: "long" };
  return new Date(`${date}T00:00`).toLocaleDateString(undefined, parts);
}

function showCameraList() {
  const list = byId("camera-list");
  const labels = status.cameras.join("\n");
  if (list.dataset.labels !== labels) {
    // Only rebuilt when the cameras change, so an open list doesn't close under the mouse.
    list.replaceChildren(...status.cameras.map((label, number) => new Option(label, number)));
    list.dataset.labels = labels;
  }
  if (document.activeElement !== list) {
    list.value = String(status.camera_number);
  }
}

function showCamera() {
  const cameraOn = status.mode !== null;
  wantPictures(cameraOn);
  // Shown until the first picture arrives.
  byId("camera-off-text").textContent = cameraOn ? "Opening the camera..." : "Camera is off";

  const badge = byId("badge");
  badge.hidden = !cameraOn;
  if (status.camera_state === "opening") {
    badge.textContent = "Opening";
    badge.className = "badge";
  } else if (status.mode === "live") {
    badge.textContent = "Live";
    badge.className = "badge live";
  } else if (status.mode === "register") {
    badge.textContent = "Registering";
    badge.className = "badge";
  }

  showRegistration(status.registration);
  if (refusal && Date.now() < refusal.until) {
    say("message", refusal.text, true);
  } else {
    refusal = null;
    say("message", status.message);
  }
  byId("error").textContent = status.error || "";
  byId("error").hidden = !status.error;
  byId("start-live").disabled = status.mode === "live";
  byId("stop").disabled = !cameraOn;
  byId("register-camera").disabled = status.mode === "register";
}

// While registering: the pose to make, the countdown, and one dot per photo.
function showRegistration(registration) {
  const countdown = byId("countdown");
  byId("registration-bar").hidden = !registration;
  countdown.hidden = true;
  if (!registration) {
    return;
  }
  const { photos_taken: taken, total, seconds_left: secondsLeft, faces_seen: faces } = registration;
  byId("pose").textContent = `Photo ${Math.min(taken + 1, total)} of ${total}: ${registration.pose}`;
  if (secondsLeft > 0) {
    countdown.textContent = secondsLeft;
    countdown.className = "countdown";
    countdown.hidden = false;
  } else if (secondsLeft === 0 && faces !== 1) {
    // The countdown is over, but the photo is only taken when exactly one face is visible.
    countdown.textContent = faces === 0 ? "No face found" : "Only one person, please";
    countdown.className = "countdown small";
    countdown.hidden = false;
  }
  const dots = Array.from({ length: total }, (_, number) => {
    const dot = document.createElement("li");
    dot.className = number < taken ? "done" : number === taken ? "current" : "";
    dot.title = `Photo ${number + 1}: ${number < taken ? "taken" : "not yet"}`;
    return dot;
  });
  byId("progress").replaceChildren(...dots);
}

// Lists are only redrawn when what they show has changed, so a button in them isn't replaced while
// it is being clicked.
function showList(listId, emptyId, rows, makeRow) {
  const list = byId(listId);
  const content = JSON.stringify(rows);
  if (list.dataset.content !== content) {
    list.replaceChildren(...rows.map(makeRow));
    list.dataset.content = content;
  }
  list.hidden = rows.length === 0;
  byId(emptyId).hidden = rows.length > 0;
}

// A list row: the name (with an extra class, like "known", for its colour), then any other parts
// (text or elements) after it.
function row(name, nameClass, ...parts) {
  const item = document.createElement("li");
  const nameText = document.createElement("span");
  nameText.className = `name ${nameClass}`;
  nameText.textContent = name; // textContent, never innerHTML: a name is shown as text, not run as HTML
  nameText.title = name; // the whole name, if the list is too narrow for it
  item.append(nameText, ...parts);
  return item;
}

function detail(text, className = "") {
  const part = document.createElement("span");
  part.className = `detail ${className}`;
  part.textContent = text;
  return part;
}

function photos(count) {
  return `${count} photo${count === 1 ? "" : "s"}`;
}

function showPresentToday() {
  byId("present-count").textContent = status.present_today.length;
  byId("today-csv").href = `/attendance.csv?date=${status.today}`;
  showList("present-list", "present-empty", status.present_today, (arrival) =>
    row(arrival.name, "", detail(arrival.time))
  );
}

function showPeople() {
  byId("people-count").textContent = status.people.length;
  showList("people-list", "people-empty", status.people, (person) => {
    const fewPhotos = person.photos < 3;
    const remove = document.createElement("button");
    remove.className = "remove";
    remove.title = `Delete ${person.name} and their photos`;
    remove.setAttribute("aria-label", remove.title);
    remove.textContent = "×";
    remove.addEventListener("click", () => deletePerson(person.name));
    const count = detail(`${photos(person.photos)}${fewPhotos ? ", add more" : ""}`, fewPhotos ? "warn" : "");
    return row(person.name, "", count, remove);
  });
}

// ---------------------------------------------------------------- the camera picture

// While the camera is on, the page asks for the newest picture, waits until it has arrived, and
// then asks for the next one. So a slow computer never piles up pictures it can't show in time.
const video = byId("video");
let picturesWanted = false;
let pictureRequested = false;

function askForPicture() {
  if (!picturesWanted) {
    pictureRequested = false;
    return;
  }
  pictureRequested = true;
  video.src = `/api/frame.jpg?time=${Date.now()}`; // a new address each time, so the browser doesn't reuse an old picture
}

video.addEventListener("load", () => {
  byId("video-box").classList.add("has-picture");
  askForPicture();
});
// No picture yet (the camera is still opening): try again a moment later.
video.addEventListener("error", () => setTimeout(askForPicture, 250));

function wantPictures(cameraOn) {
  picturesWanted = cameraOn;
  if (!cameraOn) {
    byId("video-box").classList.remove("has-picture");
  } else if (!pictureRequested) {
    askForPicture();
  }
}

// ---------------------------------------------------------------- the camera's buttons

byId("start-live").addEventListener("click", () => act("/api/live/start"));
byId("stop").addEventListener("click", () => act("/api/stop"));
byId("refresh-cameras").addEventListener("click", () => act("/api/cameras/refresh"));
byId("camera-list").addEventListener("change", (event) => {
  act("/api/camera", { camera: Number(event.target.value) });
});

// ---------------------------------------------------------------- adding a person

byId("register-camera").addEventListener("click", () => {
  say("add-message", "");
  act("/api/register/start", { name: byId("person-name").value }, "add-message");
  byId("camera").scrollIntoView({ behavior: "smooth", block: "nearest" }); // in a small window it can be out of sight
});

const personPhotos = byId("person-photos");

// Next to the button: how many photos are chosen.
function showChosenPhotos() {
  const count = personPhotos.files.length;
  byId("person-photos-text").textContent = count ? `${photos(count)} chosen` : "3 to 5 photos, each showing only them";
}
personPhotos.addEventListener("change", showChosenPhotos);

byId("register-upload").addEventListener("click", async () => {
  if (!personPhotos.files.length) {
    say("add-message", "Choose the person's photos first.", true);
    return;
  }
  const form = new FormData();
  form.append("name", byId("person-name").value);
  for (const photo of personPhotos.files) {
    form.append("photos", photo);
  }
  say("add-message", "Registering...");
  try {
    const answer = await ask("/api/people/upload", { form });
    say("add-message", answer.message);
    personPhotos.value = "";
    showChosenPhotos();
  } catch (problem) {
    say("add-message", problem.message, true);
  }
});

byId("rebuild").addEventListener("click", async () => {
  say("add-message", "Registering everyone in the photo folders...");
  try {
    say("add-message", (await ask("/api/people/rebuild", { json: {} })).message);
  } catch (problem) {
    say("add-message", problem.message, true);
  }
});

// The answer is shown under the list of registered people, and goes away by itself after a while.
let peopleMessageTimer = null;

async function deletePerson(name) {
  if (!confirm(`Delete ${name} and all their photos? This can't be undone.`)) {
    return;
  }
  try {
    sayOrHide("people-message", (await ask("/api/people/delete", { json: { name } })).message);
  } catch (problem) {
    sayOrHide("people-message", problem.message, true);
  }
  clearTimeout(peopleMessageTimer);
  peopleMessageTimer = setTimeout(() => sayOrHide("people-message", ""), 8000);
}

// ---------------------------------------------------------------- recognizing a photo

// It starts as soon as a photo is chosen.
byId("photo-file").addEventListener("change", async (event) => {
  const chooser = event.target;
  if (!chooser.files.length) {
    return;
  }
  const form = new FormData(byId("photo-form"));
  chooser.value = ""; // so that choosing the same photo again recognizes it again
  say("photo-message", "Looking for faces...");
  try {
    const result = await ask("/api/recognize", { form });
    byId("photo-image").src = result.image;
    // Each face found: its name (green when registered, red when unknown) and its similarity.
    byId("photo-faces").replaceChildren(
      ...result.faces.map((face) => row(face.name, face.known ? "known" : "unknown", detail(face.similarity.toFixed(2))))
    );
    byId("photo-caption").textContent = `${result.file_name}: a full-size copy is saved in ${result.saved_as}.`;
    byId("photo-result").hidden = false;
    const found = result.faces.length;
    say("photo-message", found ? `${found} face(s) found. Click the photo to see it bigger.` : "No faces found.");
  } catch (problem) {
    say("photo-message", problem.message, true);
  }
});

// Clicking the photo shows it as big as the window allows. A click closes it again, and so does Esc.
byId("photo-image").addEventListener("click", () => {
  byId("viewer-image").src = byId("photo-image").src;
  byId("viewer").showModal();
});
byId("viewer").addEventListener("click", () => byId("viewer").close());

// ---------------------------------------------------------------- attendance history

async function showHistory(date) {
  try {
    const answer = await ask(`/api/attendance?date=${encodeURIComponent(date)}`);
    byId("history-date").value = answer.date;
    byId("history-csv").href = `/attendance.csv?date=${answer.date}`;
    byId("history-rows").replaceChildren(
      ...answer.rows.map((arrival, index) => {
        const tableRow = document.createElement("tr");
        for (const value of [index + 1, arrival.name, arrival.time]) {
          const cell = document.createElement("td");
          cell.textContent = value;
          tableRow.append(cell);
        }
        return tableRow;
      })
    );
    byId("history-empty").hidden = answer.rows.length > 0;
    // The most recent days with attendance, newest first. Clicking one shows that day.
    byId("history-dates").replaceChildren(
      ...answer.dates.slice(0, 14).map((day) => {
        const button = document.createElement("button");
        button.className = day === answer.date ? "selected" : "";
        button.textContent = dateInWords(day, true);
        button.title = day;
        button.addEventListener("click", () => showHistory(day));
        return button;
      })
    );
    byId("history-days").hidden = answer.dates.length === 0;
    sayOrHide("history-message", "");
  } catch (problem) {
    sayOrHide("history-message", problem.message, true);
  }
}
byId("history-date").addEventListener("change", (event) => showHistory(event.target.value));

// ---------------------------------------------------------------- tabs

// The open tab is also written in the address (#add, #photo, #history), so a link can open it.
function openTab(name) {
  if (!document.querySelector(`[data-panel="${name}"]`)) {
    name = "add";
  }
  for (const tab of document.querySelectorAll("[data-tab]")) {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  }
  for (const panel of document.querySelectorAll("[data-panel]")) {
    panel.hidden = panel.dataset.panel !== name;
  }
  if (name === "history") {
    showHistory(byId("history-date").value);
  }
}
for (const tab of document.querySelectorAll("[data-tab]")) {
  tab.addEventListener("click", () => {
    history.replaceState(null, "", `#${tab.dataset.tab}`);
    openTab(tab.dataset.tab);
  });
}

openTab(location.hash.slice(1));
keepStatusUpToDate();
