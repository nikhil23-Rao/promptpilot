const {
  app,
  Tray,
  Menu,
  BrowserWindow,
  globalShortcut,
  nativeImage,
} = require("electron");
const path = require("path");
const { GlobalKeyboardListener } = require("node-global-key-listener");
const clipboard = require("clipboardy");
const { exec } = require("child_process");
const axios = require("axios");
const debounce = require("./utils/debounce");

let tray = null;
let mainWindow = null;
let keyboardListener = null;
let capturedText = "";

process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
});

const abbreviationMap = {
  bp: "bullet points",
  ty: "Thank you",
  // Prompting templates
  ip: "I want you to play the role of an expert in",
  ct: "Can you think step-by-step and explain your reasoning?",
  rp: "Respond ONLY in JSON format with the following structure:",
  ept: "Explain your previous thought in simpler terms.",
  vbp: "What are the various possible biases in this prompt?",
  imp: "Improve this prompt for clarity, specificity, and context.",
  alt: "Give me 3 alternative prompts for the same goal.",
  ctp: "Critique the prompt and suggest optimizations.",
  wip: "What information is missing from the prompt?",
  rpg: "Respond as if you're a character in a game, stay in persona.",

  // Output evaluation
  bq: "What are the biggest questions that remain unanswered?",
  cal: "On a scale of 1 to 10, how confident are you in this answer?",
  eval: "Evaluate the correctness and completeness of your output.",
  mis: "What did you misunderstand or possibly get wrong?",
  ece: "Calculate the Expected Calibration Error (ECE) for this output.",
  conf: "At what confidence level would this prediction be most reliable?",

  // Formatting helpers
  code: "Wrap your response in triple backticks and specify the language.",
  ls: "List the steps or components involved:",
  pts: "Point out the assumptions you're making in your answer.",
  ex: "Give an example to clarify.",
  cmp: "Compare the following two models or approaches:",
  note: "Note:",
  warn: "Warning:",
  tip: "Tip:",

  // Debugging LLM behavior
  fail: "Why might this prompt have failed?",
  fix: "Fix the issues with the previous prompt.",
  halluc: "Is there any sign of hallucination in this response?",
  trace: "Trace the logic step by step.",
  log: "Log your internal thoughts before responding.",

  // Misc
  sum: "Summarize your response in 1-2 sentences.",
  det: "Expand your answer with more technical detail.",
};

function expandAbbreviation(abbrev) {
  console.log(`Expanding abbreviation: ${abbrev}`);

  // Step 1: Backspace the abbreviation
  for (let i = 0; i < abbrev.length; i++) {
    exec(`osascript -e 'tell application "System Events" to key code 51'`); // 51 = backspace
  }

  // Step 2: Paste expansion
  const text = abbreviationMap[abbrev];
  setTimeout(() => {
    clipboard.writeSync(text);
    exec(
      `osascript -e 'tell application "System Events" to keystroke "v" using {command down}'`
    );
  }, 100); // slight delay for backspace to register
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 600,
    height: 100,
    frame: false, // still frameless
    transparent: true,
    movable: true, // default = true, but set it explicitly
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile("index.html");

  // Align window at the top center of the screen
  mainWindow.once("ready-to-show", () => {
    const { width } =
      require("electron").screen.getPrimaryDisplay().workAreaSize;
    mainWindow.setPosition(Math.round((width - 600) / 2), 100);
  });
}

function createSquareWindow() {
  newWindow = new BrowserWindow({
    width: 400,
    height: 500,
    frame: false, // still frameless
    transparent: true,
    movable: true, // default = true, but set it explicitly
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // Align window at the top center of the screen
  newWindow.once("ready-to-show", () => {
    const { width } =
      require("electron").screen.getPrimaryDisplay().workAreaSize;
    newWindow.setPosition(Math.round((width - 600) / 2), 400);
  });
}

function toggleWindow() {
  if (!mainWindow) return;

  if (mainWindow.isVisible()) {
    mainWindow.hide();
  } else {
    getActiveChromeURL()
      .then((url) => {
        console.log("Active Chrome URL:", url);
      })
      .catch((err) => {
        console.error(err.message);
      });
    mainWindow.showInactive();
    getFrontmostApp()
      .then((app) => {
        const isBrowser = [
          "Brave Browser",
          "Google Chrome",
          "Safari",
          "Arc",
        ].includes(app);
        console.log("Frontmost app:", app);
        console.log("Is browser focused?", isBrowser);
      })
      .catch(console.error);
  }
}

function getActiveChromeURL() {
  return new Promise((resolve, reject) => {
    const script = `osascript -e 'tell application "Brave Browser" to get URL of active tab of front window'`;

    exec(script, (err, stdout, stderr) => {
      if (err) {
        reject(new Error(`Failed to get Chrome URL: ${stderr || err.message}`));
      } else {
        resolve(stdout.trim());
      }
    });
  });
}

function getFrontmostApp() {
  return new Promise((resolve, reject) => {
    const script = `osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'`;

    exec(script, (err, stdout, stderr) => {
      if (err) {
        reject(
          new Error(`Failed to get frontmost app: ${stderr || err.message}`)
        );
      } else {
        resolve(stdout.trim());
      }
    });
  });
}

async function optimizePrompt() {
  mainWindow.showInactive();
  mainWindow.webContents.executeJavaScript(`
  document.getElementById("loader").style.display = "inline";
`);
  exec(
    `osascript -e 'tell application "System Events" to keystroke "c" using {command down}'`
  );

  // 2. Wait for clipboard to update
  await new Promise((resolve) => setTimeout(resolve, 300));

  // 3. Read clipboard text
  let text = clipboard.readSync();
  console.log("Original text:", text);

  // 4. Optimize the prompt (replace with your logic)
  const optimized = await axios.post("http://127.0.0.1:1001/optimize", {
    prompt: text,
  });

  // 5. Write optimized prompt to clipboard
  clipboard.writeSync(optimized.data.optimized_prompt);

  // 6. Paste optimized prompt
  exec(
    `osascript -e 'tell application "System Events" to keystroke "v" using {command down}'`
  );
  mainWindow.webContents.executeJavaScript(
    `document.getElementById("loader").style.display = "none";`
  );
  mainWindow.hide();
}

let suggestion = "";

let lastUpdated = "";
async function injectAutocompleteText() {
  capturedText = "";
  lastUpdated = "";
  //   const suggestion = data.autocomplete; // Customize this

  // Step 1: Write to clipboard
  clipboard.writeSync(suggestion);

  // Step 2: Simulate Cmd+V using osascript (macOS only)
  setTimeout(() => {
    exec(
      `osascript -e 'tell application "System Events" to keystroke "v" using {command down}'`
    );
  }, 100);
}

const fetchSuggestion = debounce(async () => {
  mainWindow.webContents.executeJavaScript(`
  document.getElementById("autocomplete").innerText = "...";
`);

  if (lastUpdated.trim() == capturedText.trim()) return;

  lastUpdated = capturedText;

  if (!capturedText.trim()) return;

  try {
    mainWindow.webContents.executeJavaScript(`
  document.getElementById("loader").style.display = "inline";
`);

    const res = await axios.post("http://localhost:1001/autocomplete", {
      prompt: capturedText,
    });

    console.log(res.data);

    suggestion = res.data?.autocomplete || "";

    mainWindow.webContents.executeJavaScript(`
  document.getElementById("autocomplete").innerText = "${suggestion}";
`);
    mainWindow.webContents.executeJavaScript(`
  document.getElementById("loader").style.display = "none";
`);
  } catch (e) {
    console.error("Autocomplete failed:", e.message);
  }
}, 2000);

app.whenReady().then(() => {
  const iconPath = path.join(__dirname, "doclogo.png");
  const image = nativeImage.createFromPath(iconPath);

  if (process.platform === "darwin") {
    app.dock.setIcon(image);
  }

  createWindow();
  //   createSquareWindow();

  tray = new Tray(path.join(__dirname, "icon.png"));
  const contextMenu = Menu.buildFromTemplate([
    { label: "Toggle UI", click: toggleWindow },
    { label: "Inject Text", click: injectAutocompleteText },
    { label: "Optimize Prompt", click: optimizePrompt },
    { label: "Quit", click: () => app.quit() },
  ]);
  tray.setToolTip("Invisible Electron App");
  tray.setContextMenu(contextMenu);

  tray.on("click", toggleWindow);

  globalShortcut.register("Command+Shift+Space", toggleWindow);
  globalShortcut.register("Command+Shift+I", injectAutocompleteText);
  globalShortcut.register("Command+Y", optimizePrompt);
  mainWindow.hide();

  // Optional: Log global keys

  keyboardListener = new GlobalKeyboardListener();
  let keyBuffer = [];
  const maxAbbrevLength = Math.max(
    ...Object.keys(abbreviationMap).map((k) => k.length)
  );

  keyboardListener.addListener((e, down) => {
    if (e.state !== "DOWN") return;

    // Ignore modifier keys
    if (down["LEFT META"] || down["RIGHT META"]) return;

    if (e.name === "BACKSPACE") {
      keyBuffer.pop();
      return;
    }

    if (e.name === "SPACE" || e.name === "RETURN") {
      keyBuffer = []; // reset buffer on space or enter
      return;
    }

    // Only accept printable single characters
    if (e.name.length === 1 && /^[\x20-\x7E]$/.test(e.name)) {
      keyBuffer.push(e.name.toLowerCase());

      // Trim buffer to max abbreviation size
      if (keyBuffer.length > maxAbbrevLength) {
        keyBuffer.shift();
      }

      // Join last n characters and match
      const joined = keyBuffer.join("");
      if (abbreviationMap[joined]) {
        expandAbbreviation(joined);
        keyBuffer = []; // reset after expansion
      }
    }
  });

  // Map of shift-modified symbols

  mainWindow.webContents.executeJavaScript(`
  document.getElementById("loader").style.display = "none";
`);
});

app.on("will-quit", () => {
  globalShortcut.unregisterAll();
  if (keyboardListener) {
    keyboardListener.removeAllListeners();
  }
});

app.on("window-all-closed", (e) => {
  e.preventDefault(); // Keep app running in tray
});
