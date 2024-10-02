# Characterizing Context Influence and Hallucination In Summarization

This is the official repository for the work "Characterizing Context Influence and Hallucination In Summarization." 

## Environment Setup
We used Python 3.10.12 in our implementation. Run the following lines to set up the evironment: 

```bash
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install -r requirements.txt
``` 

We also used [AlignScore](https://github.com/yuh-zha/AlignScore) in our evalutions. Here's the command to properly add our repository with AlignScore:

```bash
git submodule init
git submodule update
```