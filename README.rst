mdsf2018-prod
=============
External hyperlinks, like Python_.
.. _Python: http://www.python.org/

This is a demonstration Code refactoring to enable models to go to production.
It is based on the winner solution of the Meilleur data scientist de France competition.

Setup with conda
----------------

Setup your environment with conda: 
    
    # conda env create -f env.yml
    
Activate the newly created conda environment:

    # conda activate mdsf2018

(Optional) Add your the conda environment as a jupyter kernel
Your conda environment must be activated.

    # python -m ipykernel install --user --name mdsf2018 --display-name mdsf2018


Run
---

Command line:

    # python -m mdsf2018
    
Jupyter notebook/Jupyter lab:

* Open the jupyter notebook
* Set your kernel to mdsf2018
* Enjoy

Configuration
-------------

If you want to submit a solution, you will have to sign up and get your token from the meilleur data scientist de France competition website.

.. winner: https://github.com/NikitaLukashev/MDF-2018
.. Meilleur data scientist de France: https://www.meilleurdatascientistdefrance.com/
.. meilleur data scientist de France competition: https://qscore.meilleurdatascientistdefrance.com/competitions/32153fb0-4a40-4579-bb7c-c61cdd8ee9a9/info