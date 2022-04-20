# S22. Sensors and Sensing

## Home Assignments

### HA 1
* [code](./HA1/HA1.ipynb)
* [report](./HA1/HA1.pdf) / <a href="https://www.mathcha.io/editor/kx1P4c9liqOH2xHD2wmxjFLnPqEyuv6XJzOUVdEoxP"><img src="https://cdn.mathcha.io/resources/logo.png" width="20" title="hover text"></a>

## Project setup
* Clone the repository
```sh
git clone https://github.com/br4ch1st0chr0n3/SS
```

* Open it in VS Code with Python extension and Jupyter kernel
```sh
code ./SS
```

* Install [Miniconda](https://conda.io/en/latest/miniconda.html)

* In VS Code, press `Ctrl` + `` ` `` to open a terminal

* Create an [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
```sh
conda env create -f env.yml
conda activate ss_env
```

* Open the desired `.ipynb` and click `Run All` at the top.

* Select the `ss_env` in the list of environments and wait until all cells finish running.

* If you no more need this environment, [remove it](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment).
```sh
conda deactivate
conda remove --name ss_env --all
```