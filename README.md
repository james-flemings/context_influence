# A Characterization of Hallucination and Privacy with Contextual Privacy-Aware Decoding

This is the official repository for the work "A Characterization of Hallucination and Privacy with Contextual Privacy-Aware Decoding." In this work, we analytically connect hallucination with privacy then experimentally evaluate this connection.  

## Environment Setup
We used Python 3.9.2 in our implementation. Run the following lines to set up the evironment: 

```bash
sudo apt install python3.9
sudo apt install python3.9-venv
python3.9 -m ensurepip --upgrade
python3.9 -m venv venv
source venv/bin/activate
python3.9 -m pip install -r requirements.txt
``` 

We also used [AlignScore](https://github.com/yuh-zha/AlignScore) in our evalutions. Here's the command to properly add our repository with AlignScore:

```bash
git submodule init
git submodule update
```
