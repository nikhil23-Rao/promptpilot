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

  mainWindow.once("ready-to-show", () => {
    const { width } =
      require("electron").screen.getPrimaryDisplay().workAreaSize;
    mainWindow.setPosition(Math.round((width - 400) / 2), 0);
  });

  mainWindow.on("blur", () => {});
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
    { label: "Quit", click: () => app.quit() },
  ]);
  tray.setToolTip("Invisible Electron App");
  tray.setContextMenu(contextMenu);

  tray.on("click", toggleWindow);

  globalShortcut.register("Command+Shift+Space", toggleWindow);
  globalShortcut.register("Command+Shift+I", injectAutocompleteText);

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
