const { app, Tray, Menu, BrowserWindow, globalShortcut } = require("electron");
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
    width: 400,
    height: 60,
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

  // Align window at the bottom center of the screen
  mainWindow.once("ready-to-show", () => {
    const { width, height } =
      require("electron").screen.getPrimaryDisplay().workAreaSize;
    mainWindow.setPosition(Math.round((width - 400) / 2), height - 60);
  });
}

function toggleWindow() {
  if (!mainWindow) return;

  if (mainWindow.isVisible()) {
    mainWindow.hide();
  } else {
    mainWindow.show();
    mainWindow.focus();
  }
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
      console.log(`Key pressed: ${e.name}`);
    }
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
