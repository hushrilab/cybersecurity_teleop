# Cybersecurity for Teleoperation Robot

## Run Script

Please make sure to have Anaconda installed.

In the project directory, run:

#### `conda env create -f cybersecurity_robotics.yaml`

to create the virtual environment for the project, 

#### `conda activate cybersecurity_robotics`

to activate the virtual environment, 

Then run:

#### `jupyter-notebook`

to launch Jupyter Notebook.

## To Do

- Account for when total packets in or out is 0

## Acknowledgements

The code implemented in this repo for converting pcap file to csv is based on code in the repository https://github.com/dmbb/Protozoa/tree/master.

Original work:

@inproceedings{protozoa,
  title={Poking a Hole in the Wall: Efficient Censorship-Resistant Internet Communications by Parasitizing on WebRTC},
  author={Barradas, Diogo and Santos, Nuno and Rodrigues, Lu{\'i}s and Nunes, V{\'i}tor},
  booktitle={Proceedings of the ACM SIGSAC Conference on Computer and Communications Security},
  year={2020},
  address={Virtual Event, USA}
}
