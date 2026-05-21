Gemini said
Here is the command to convert a Python script to a Jupyter Notebook using Jupytext:

```Bash
jupytext --to ipynb notebooks/release-demo.py
```

This will create a new file named your_script.ipynb in the same directory.

Alternative: Pair and sync the files
If you want to keep both files synced so that changes in one automatically reflect in the other, you can set them up as a paired notebook:

```Bash
jupytext --set-formats ipynb,py your_script.py
```

Would you like me to explain how to configure Jupytext to automatically pair all notebooks in a specific directory?
