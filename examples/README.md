## Requirements                                                                                                                                                                                                     
Each process was ran with Python3.11. See the provided [requirements file](requirements.txt) for each the Python requirements.

## Usage
```shell
$ python3.11 <script>.py
```

Each python script was made as simple as possible such that just to utilize any script, simply invoke Python 3.11 followed by the name of the Python script (e.g., <script>.py). The process entails no distinctive steps beyond running the Python interpreter and specifying the script file (except when wanting to change already set hyperparameters).

-------
To utilize the C++ scripts to simulate data with and without anomolies for the AutoEncoder, it is necessary to make sure the appropraite libraires for ROOT are available in the environment. If so, then the following command can produce the desired normalData.root/mixedData.root files with the appropriate simulated data. Double check 'png' files for output of the simulations.

This will generate the normalData.root file to train the AutoEncoder on:
```shell
$ g++ -o generate_normal_data generate_normal_data.cpp `root-config --cflags --glibs`
$ ./generate_normal_data

```
This will generate the mixedData.root file with anomolies injected to be used to test (optional):
```shell
$ g++ -o generate_anom_data generate_anom_data.cpp `root-config --cflags --glibs`
$ ./generate_normal_data

```

-------
