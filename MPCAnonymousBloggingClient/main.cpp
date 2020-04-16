#include <iostream>
#include "Client.h"

int main(int argc, char* argv[]) {

    CmdParser parser;
    auto parameters = parser.parseArguments("", argc, argv);

    string fieldType = parser.getValueByKey(parameters, "fieldType");
    cout<<"fieldType = "<<fieldType<<endl;

    if(fieldType.compare("ZpMersenne31") == 0)
    {
        Client<ZpMersenneIntElement> client(argc, argv);
        auto t1 = high_resolution_clock::now();
        client.run();

        vector<ZpMersenneIntElement> msg, unitVector;
        ZpMersenneIntElement e;
        auto t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2-t1).count();
        cout << "time in milliseconds: " << duration << endl;
        cout << "./end main" << '\n';

    }
    else if(fieldType.compare("ZpMersenne61") == 0)
    {

        Client<ZpMersenneLongElement> client(argc, argv);
        auto t1 = high_resolution_clock::now();
        client.run();

        vector<ZpMersenneLongElement> msg, unitVector;
        ZpMersenneLongElement e;
        client.readServerFile("server0ForClient0inputs.txt", msg, unitVector, &e);
        client.checkServerFiles(0);

        client.checkExtractMsgs();
        auto t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2-t1).count();
        cout << "time in milliseconds: " << duration << endl;
        cout << "end main" << '\n';

    }

    return 0;
}
