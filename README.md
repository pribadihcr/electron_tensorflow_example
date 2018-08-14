## Electron with Tensorflow example

### ENVIRONMENTS
- Ubuntu 16.04 LTS
- Anaconda 3.6.4

Install necessary python packages
```text
pip install zerorpc
pip install pyinstaller
pip install tensorflow
```

Clean the caches
```text
rm -rf ~/.node-gyp
rm -rf ~/.electron-gyp
rm -rf ./node_modules
```

Run npm
```text
npm install --runtime=electron --target=2.0.2
```

Testing before build the package. Open new terminal
```text
python pyDL/api.py
```
Open another terminal terminal
```text
./node_modules/.bin/electron .
```
Try click button. If it shows Tensorflow: 1.8 then succeed, else what is the error (make issue)

If testing above succeed, we can make a package.
Python executable
```text
pyinstaller pyDL/api.py --distpath pyDLdist

rm -rf build/
rm -rf api.spec
```

Install electron-rebuild to avoid 'NODE_MODULE_VERSION' error
```text
Install electron-rebuild
./node_modules/.bin/electron-rebuild
```

Package the app
```text
./node_modules/.bin/electron-packager . --overwrite --ignore="pyDL$"
```

Execute the app
```text
./electron_TF_example-linux-x64/electron_TF_example-linux-x64
```
If you still see any error please report/write in this readme. 





