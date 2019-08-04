#!/usr/bin/env bash

cd ../MPCAnonymousBloggingClient && ./MPCAnonymousBloggingClient -server ${1} -numServers ${2} -numClients ${3} \
   -fieldType ZpMersenne31 -l ${4}

cd ../MPCAnonymousBlogging && ./MPCAnonymousBloging -partyID ${1} -numServers ${2} -numClients ${3} \
   -fieldType ZpMersenne31 -l ${4} -partiesFile parties.conf  -internalIterationsNumber 1 -numThreads ${5}
