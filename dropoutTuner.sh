#!/bin/bash
for i in {3..10}
    do
        #Modifies the 65th line of CNetworks.py
        sed -i "65s/.*/    dropoutProb=${i}/10  /" CNetworks.py
        #Run the main function model
        python CRnn.py
    done