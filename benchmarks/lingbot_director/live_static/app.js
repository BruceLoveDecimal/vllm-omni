// Browser side of the live Director loop: plays streamed frames, draws event cards on
// tracked objects, and sends keys, clicks and text events back to the orchestrator.
"use strict";

const CAMERA_KEYS = new Set(["w", "a", "s", "d", "i", "j", "k", "l"]);
const HOTKEYS = ["u", "o", "f", "g"];
const SLOTS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
const HOTKEY_LABEL = { u: "角色动作", o: "角色动作", f: "环境变化", g: "环境变化" };

const screen = document.getElementById("screen");
const context = screen.getContext("2d");
const overlay = document.getElementById("overlay");
const menu = document.getElementById("menu");

let socket = null;
let state = { objects: [], proposals: {}, slots: {} };
// Boxes measured on a given chunk, so a card follows the frame actually on screen.
const boxesByChunk = new Map();
const frameQueue = [];
let pendingChunk = 0;
// Bitmaps decode asynchronously; chaining keeps them in arrival order.
let decoding = Promise.resolve();
const STATUS = { sent: "等待生效", playing: "进行中", verifying: "核验中" };
const RUNG = { prompt: "提示词", rewrite: "改写场景", reanchor: "重新锚定" };
let shownChunk = 0;
let fps = 16;

function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
  }
}

function connect() {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onmessage = (event) => {
    if (typeof event.data !== "string") {
      const chunk = pendingChunk;
      const blob = new Blob([event.data], { type: "image/jpeg" });
      decoding = decoding
        .then(() => createImageBitmap(blob))
        .then((bitmap) => frameQueue.push({ bitmap, chunk }));
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === "frames") {
      pendingChunk = message.chunk;
      fps = message.fps;
    } else if (message.type === "state") {
      onState(message);
    }
  };
  socket.onclose = () => {
    document.getElementById("status").textContent = "连接断开，重连中";
    setTimeout(connect, 1000);
  };
}

function onState(message) {
  state = message;
  document.getElementById("status").textContent = message.running ? "运行中" : "未开始";
  if (!message.running) {
    return;
  }
  boxesByChunk.set(message.objects_chunk, message.objects);
  document.getElementById("clock").textContent = `第 ${message.chunk} 段`;
  renderPanel();
}

function renderPanel() {
  const proposals = document.getElementById("proposals");
  proposals.replaceChildren(
    ...HOTKEYS.filter((key) => state.proposals[key]).map((key) =>
      cardButton(key, state.proposals[key], HOTKEY_LABEL[key], () => send({ type: "hotkey", key }))
    )
  );

  const slots = document.getElementById("slots");
  slots.replaceChildren(
    ...SLOTS.filter((slot) => state.slots[slot]).map((slot) =>
      cardButton(slot, state.slots[slot], "槽位", () => send({ type: "slot", slot }))
    )
  );

  const scheduler = state.scheduler;
  const active = scheduler.active;
  document.getElementById("active").textContent = active
    ? `${active.request}（${STATUS[active.status] ?? active.status}，${RUNG[active.rung] ?? active.rung}，剩 ${active.remaining} 段，来源 ${active.source}）`
    : "无";
  document.getElementById("pending").replaceChildren(
    ...state.pending.map((item) => textLine("pending", `推理中：${item.request}`))
  );
  document.getElementById("queue").replaceChildren(
    ...scheduler.queue.map((item) => textLine("pending", `排队：${item.request}`))
  );
  document.getElementById("scene").textContent = scheduler.scene;
  document.getElementById("facts").replaceChildren(...scheduler.facts.map((fact) => listItem(fact)));
  document.getElementById("prompt").textContent = scheduler.prompt;
  document.getElementById("notices").replaceChildren(
    ...state.notices
      .slice()
      .reverse()
      .map((notice) => listItem(`第 ${notice.chunk} 段 未执行：${notice.text}`, notice.level))
  );
  for (const element of document.querySelectorAll("kbd[data-key]")) {
    element.classList.toggle("held", state.camera.includes(element.dataset.key));
  }
}

function cardButton(key, text, label, onClick) {
  const button = document.createElement("button");
  button.className = "card";
  const kbd = document.createElement("kbd");
  kbd.textContent = key.toUpperCase();
  const body = document.createElement("span");
  body.textContent = `${text}`;
  body.title = label;
  button.append(kbd, body);
  button.onclick = onClick;
  return button;
}

function textLine(className, text) {
  const div = document.createElement("div");
  div.className = className;
  div.textContent = text;
  return div;
}

function listItem(text, className) {
  const item = document.createElement("li");
  item.textContent = text;
  if (className) {
    item.className = className;
  }
  return item;
}

function boxesFor(chunk) {
  let best = null;
  for (const [measured, objects] of boxesByChunk) {
    if (measured <= chunk && (best === null || measured > best)) {
      best = measured;
    }
  }
  for (const measured of boxesByChunk.keys()) {
    if (best !== null && measured < best) {
      boxesByChunk.delete(measured);
    }
  }
  return best === null ? [] : boxesByChunk.get(best);
}

// Cards are kept per object and moved in place: replacing them as the boxes update would
// swap a button out from under a click.
const cards = new Map();

function renderCards() {
  const scale = screen.clientWidth / screen.width;
  const objects = boxesFor(shownChunk);
  const seen = new Set();
  for (const object of objects) {
    seen.add(object.object_id);
    let entry = cards.get(object.object_id);
    if (!entry) {
      const target = document.createElement("div");
      target.className = "target";
      const card = document.createElement("div");
      card.className = "event-card";
      overlay.append(target, card);
      entry = { target, card, content: "" };
      cards.set(object.object_id, entry);
    }
    const content = [object.name, ...object.actions].join("\n");
    if (entry.content !== content) {
      const name = document.createElement("b");
      name.textContent = object.name;
      entry.card.replaceChildren(name, ...object.actions.map((action) => actionChip(object, action)));
      entry.content = content;
    }
    const [x0, y0, x1, y1] = object.box;
    Object.assign(entry.target.style, {
      left: `${x0 * scale}px`,
      top: `${y0 * scale}px`,
      width: `${(x1 - x0) * scale}px`,
      height: `${(y1 - y0) * scale}px`,
    });
    Object.assign(entry.card.style, { left: `${x0 * scale}px`, top: `${y0 * scale}px` });
  }
  for (const [objectId, entry] of cards) {
    if (!seen.has(objectId)) {
      entry.target.remove();
      entry.card.remove();
      cards.delete(objectId);
    }
  }
}

function actionChip(object, action) {
  const chip = document.createElement("button");
  chip.className = "chip";
  chip.textContent = action;
  chip.onclick = (event) => {
    event.stopPropagation();
    menu.hidden = true;
    send({ type: "object_action", object_id: object.object_id, action });
  };
  return chip;
}

// Interaction latency matters more than smoothness: past this backlog, skip ahead.
const MAX_BUFFER_SECONDS = 2;
const CATCH_UP_SECONDS = 1;

function play() {
  if (frameQueue.length > MAX_BUFFER_SECONDS * fps) {
    for (const skipped of frameQueue.splice(0, frameQueue.length - CATCH_UP_SECONDS * fps)) {
      skipped.bitmap.close();
    }
  }
  if (frameQueue.length > 0) {
    const frame = frameQueue.shift();
    context.drawImage(frame.bitmap, 0, 0, screen.width, screen.height);
    frame.bitmap.close();
    shownChunk = frame.chunk;
    renderCards();
  }
  document.getElementById("buffer").textContent = `缓冲 ${(frameQueue.length / fps).toFixed(1)} 秒`;
}

function typing(event) {
  return ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
}

window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (typing(event) || event.repeat || event.metaKey || event.ctrlKey) {
    return;
  }
  if (CAMERA_KEYS.has(key)) {
    send({ type: "key", key, down: true });
  } else if (HOTKEYS.includes(key)) {
    send({ type: "hotkey", key });
  } else if (SLOTS.includes(key)) {
    send({ type: "slot", slot: key });
  }
});

window.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  if (CAMERA_KEYS.has(key)) {
    send({ type: "key", key, down: false });
  }
});

// Releasing every key when the page loses focus keeps the camera from drifting on.
window.addEventListener("blur", () => {
  for (const key of CAMERA_KEYS) {
    send({ type: "key", key, down: false });
  }
});

screen.addEventListener("click", (event) => {
  const rect = screen.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * screen.width;
  const y = ((event.clientY - rect.top) / rect.height) * screen.height;
  const under = boxesFor(shownChunk).filter(
    (object) => object.box[0] <= x && x <= object.box[2] && object.box[1] <= y && y <= object.box[3]
  );
  if (under.length === 0) {
    menu.hidden = true;
    return;
  }
  // The smallest box wins, so a handle is picked over the door around it.
  const area = (object) => (object.box[2] - object.box[0]) * (object.box[3] - object.box[1]);
  const object = under.reduce((best, item) => (area(item) < area(best) ? item : best));
  const title = document.createElement("b");
  title.textContent = object.name;
  menu.replaceChildren(title, document.createElement("br"), ...object.actions.map((action) => actionChip(object, action)));
  Object.assign(menu.style, { left: `${event.clientX - rect.left}px`, top: `${event.clientY - rect.top}px` });
  menu.hidden = false;
});

document.getElementById("text-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("text");
  if (input.value.trim()) {
    send({ type: "text", text: input.value.trim() });
    input.value = "";
  }
});

const slotKey = document.getElementById("slot-key");
slotKey.replaceChildren(
  ...SLOTS.map((slot) => {
    const option = document.createElement("option");
    option.value = slot;
    option.textContent = slot;
    return option;
  })
);

document.getElementById("slot-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = document.getElementById("slot-text");
  if (text.value.trim()) {
    send({ type: "slot_set", slot: slotKey.value, text: text.value.trim() });
    text.value = "";
  }
});

document.getElementById("start").onclick = () => send({ type: "start" });
document.getElementById("stop").onclick = () => send({ type: "stop" });

function loop() {
  play();
  setTimeout(loop, 1000 / fps);
}

connect();
loop();
