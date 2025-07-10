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

let tray = null;
let mainWindow = null;
let keyboardListener = null;

process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise, "reason:", reason);
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 600, // Match toolbar width
    height: 100,
    show: true,
    frame: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    transparent: true,
    hasShadow: false,
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
    mainWindow.setPosition(Math.round((width - 600) / 2), 90);
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
  // 1. Simulate Cmd+C (copy)
  exec(
    `osascript -e 'tell application "System Events" to keystroke "c" using {command down}'`
  );

  // 2. Wait for clipboard to update
  await new Promise((resolve) => setTimeout(resolve, 300));

  // 3. Read clipboard text
  let text = clipboard.readSync();
  console.log("Original text:", text);

  // 4. Optimize the prompt (replace with your logic)
  const optimized = text.toUpperCase(); // Example optimization

  // 5. Write optimized prompt to clipboard
  clipboard.writeSync(optimized);

  // 6. Paste optimized prompt
  exec(
    `osascript -e 'tell application "System Events" to keystroke "v" using {command down}'`
  );
}

function injectAutocompleteText() {
  const suggestion = "This is your autocomplete suggestion"; // Customize this

  // Step 1: Write to clipboard
  clipboard.writeSync(suggestion);

  // Step 2: Simulate Cmd+V using osascript (macOS only)
  setTimeout(() => {
    exec(
      `osascript -e 'tell application "System Events" to keystroke "v" using {command down}'`
    );
  }, 300);
}

app.whenReady().then(() => {
  const iconPath = path.join(__dirname, "doclogo.png");
  const image = nativeImage.createFromPath(iconPath);

  if (process.platform === "darwin") {
    app.dock.setIcon(image);
  }

  createWindow();

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

  // Optional: Log global keys
  keyboardListener = new GlobalKeyboardListener();
  keyboardListener.addListener((e) => {
    if (e.state === "DOWN") {
    }
  });
  let capturedText = "";
  let shiftDown = false;
  let capsLockOn = false;

  // Map of shift-modified symbols

  keyboardListener.addListener((e) => {
    const key = e.name;
    const state = e.state;

    // Track shift state
    if (key === "Left Shift" || key === "Right Shift") {
      shiftDown = state === "DOWN";
      return;
    }

    // Track Caps Lock toggle
    if (key === "Caps Lock" && state === "DOWN") {
      capsLockOn = !capsLockOn;
      return;
    }

    if (state !== "DOWN") return;

    if (key === " " || key === "SPACE" || key === "Spacebar") {
      capturedText += " ";
    }
    if (key === "BACKSPACE" || key === "Delete") {
      capturedText = capturedText.slice(0, -1);
    }

    // Ignore control/meta keys (Tab, Enter, Arrows, etc.)
    if (key.length !== 1 || !/^[\x20-\x7E]$/.test(key)) return;

    // Handle letters (respect Shift and Caps Lock)
    if (/[a-zA-Z]/.test(key)) {
      const isUpper = (shiftDown && !capsLockOn) || (!shiftDown && capsLockOn);
      capturedText += isUpper ? key.toUpperCase() : key.toLowerCase();
      console.log("Captured Text:", capturedText);
      return;
    }

    // Handle numbers/symbols with shift

    console.log("Captured Text:", capturedText);
  });
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
