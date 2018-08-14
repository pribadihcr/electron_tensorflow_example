const zerorpc = require("zerorpc")
let client = new zerorpc.Client()

client.connect("tcp://127.0.0.1:4242")

client.invoke("echo", "server ready", (error, res) => {
  if(error || res !== 'server ready') {
    console.error(error)
  } else {
    console.log("server is ready")
  }
})
 
const { ipcRenderer } = require( "electron" );
document.addEventListener( "DOMContentLoaded", () => {
    const version = process.version;
    const e = document.getElementById( "info" );
    e.textContent = `I'm running Node.js version: ${ version }`;

    const btn = document.getElementById( "clickme" );
    btn.addEventListener( "click", e => {
        client.invoke("detect_yolo", "/home/deep307/SHARE/TRAFFICHAIN/POC_TrafficViolation/pyDL/data/APA-0102-CE8J4yKmTEQ.mp4",(error, msg) => {
            if(error) {
                console.error(error)
            } else {
                console.log( "Detecting traffic violation." );
                ipcRenderer.send( "show-dialog", { message: msg } );
            }
        } );
    } );
} );
