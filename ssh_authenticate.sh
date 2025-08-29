#!/bin/bash

# Start the ssh-agent
eval "$(ssh-agent -s)"

# Add the SSH private key
ssh-add ~/.ssh/id_rsa

