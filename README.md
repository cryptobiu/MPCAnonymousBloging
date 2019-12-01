### Anonymous Blogging

The repository implements the Anonymous Blogging

##### Abstract


##### Installation

The protocol written in c++ and uses c++11 standard. It uses [libscapi](https://github.com/cryptobiu/libscapi).  
For `libscapi` installation instructions, visit [here](https://github.com/cryptobiu/libscapi/blob/master/build_scripts/INSTALL.md).  
After you installed `libscapi`, run `./MATRIX/build.sh`

##### Usage

The protocol designed for at least 3 parties.
To run the the protocol open a terminal and run:  
`./run_protocol.sh <min_party_id> <max_party_id> <number_of_parties> <number_of_clients> <security_level> <parties_file> <number_of_iterations> <number_of_threads> <number_of_malicious_parties> <create_files>` 

* create_files - if set to `true`, generates client files.
* parties_file - a file that contains the ip addresses and the port of all the parties.  
An example file can be found [here](../master/Parties.txt).
    