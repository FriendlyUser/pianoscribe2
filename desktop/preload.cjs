// The only bridge between the web interface and the desktop shell.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("pianoscribe", {
  savePdf: (jobId, title) => ipcRenderer.invoke("save-pdf", String(jobId), String(title)),
  openExternal: (url) => ipcRenderer.send("open-external", String(url)),
});
