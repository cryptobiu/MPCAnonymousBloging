//
// Created by moriya on 12/3/18.
//

#ifndef MPCANONYMOUSBLOGGINGCLIENT_CLIENT_H
#define MPCANONYMOUSBLOGGINGCLIENT_CLIENT_H

#include <libscapi/include/primitives/Mersenne.hpp>
#include <libscapi/include/primitives/Matrix.hpp>
#include <libscapi/include/primitives/Prg.hpp>
#include <libscapi/include/infra/Common.hpp>
#include <libscapi/include/infra/Measurement.hpp>
#include <libscapi/include/cryptoInfra/Protocol.hpp>


template <class FieldType>
class Client {

private:
    int l;
    int numClients, sqrtR;
    int numServers;
    int T; //number of malicious servers

    // sizes are in bytes
    TemplateField<FieldType> *field;
    VDM<FieldType> matrix_vand;

    PrgFromOpenSSLAES prg;

    vector<FieldType> makeInputVector();
    vector<vector<FieldType>> createShares(vector<FieldType> & vals);
    void writeServersFiles(vector<vector<FieldType>> & shares, int clientID);

    void calcPairMessages(FieldType & a, FieldType & b, int counter);

public:
    Client(int argc, char * argv[]);
    ~Client();

    void run();

    void readServerFile(string fileName, vector<FieldType> & msg, vector<FieldType> & unitVector, FieldType * e);
    void checkServerFiles(int clientID);

    void extractMessages(vector<FieldType> & messages, vector<int> & counters, int numMsgs);
    void checkExtractMsgs();


};


template<class FieldType>
Client<FieldType>::Client(int argc, char **argv){

    CmdParser parser;
    auto arguments = parser.parseArguments("", argc, argv);

    l = stoi(parser.getValueByKey(arguments, "l"));
    numServers = stoi(parser.getValueByKey(arguments, "numServers"));
    numClients = stoi(parser.getValueByKey(arguments, "numClients"));
    string fieldType = parser.getValueByKey(arguments, "fieldType");

    sqrtR = (int)(sqrt(2.7 * numClients))+1;
    T = (numServers+1)/2 - 1;



    auto key = prg.generateKey(128);
    prg.setKey(key);

    if(fieldType.compare("ZpMersenne31") == 0) {
        field = new TemplateField<FieldType>(2147483647);
    } else if(fieldType.compare("ZpMersenne61") == 0) {
        field = new TemplateField<FieldType>(0);
    } else if(fieldType.compare("ZpKaratsuba") == 0) {
        field = new TemplateField<FieldType>(0);
    } else if(fieldType.compare("GF2E") == 0) {
        field = new TemplateField<FieldType>(8);
    } else if(fieldType.compare("Zp") == 0) {
        field = new TemplateField<FieldType>(2147483647);
    }

    matrix_vand.allocate(numServers,numServers,field);
    matrix_vand.InitVDM();
cout<<"end ctor"<<endl;

}

template <class FieldType>
Client<FieldType>::~Client()
{
    delete field;

}

template<class FieldType>
void Client<FieldType>::run() {

    for(int i=0; i<numClients; i++) {


        auto vals = makeInputVector();

        auto shares = createShares(vals);

        writeServersFiles(shares, i);
    }

}

template<class FieldType>
vector<FieldType> Client<FieldType>::makeInputVector(){

    vector<FieldType> msg(l);
    auto r = field->Random();
    msg[l-1] = r;
    for (int k=0; k<l-1; k++){
        msg[k] = field->Random() + r;
    }

    //Choose random indices i,j
    int i = getRandomInRange(0, sqrtR-1, &prg);
    int j = getRandomInRange(0, sqrtR-1, &prg);

    cout<<"i = "<<i<< " j = "<<j<<endl;

    vector<FieldType> vals(2*(l*sqrtR + sqrtR) + 1, *field->GetZero());

    //Set the vector [0, 0, ..., msg, 0, ..., 0]
    for (int m=0; m<l; m++){
        vals[l* i + m] = msg[m];
    }

    //Set the vector [0, 0, ..., msg^2, 0, ..., 0]
    for (int m=0; m<l; m++){
        vals[l*sqrtR + l* i + m] = msg[m] * msg[m];
    }

    //Set the one in the first [0, 0, ..., 1, 0, ..., 0] vector
    vals[2*l*sqrtR + i] = *field->GetOne();

    //Set the one in the second [0, 0, ..., 1, 0, ..., 0] vector
    vals[2*l*sqrtR + sqrtR + j] = *field->GetOne();

    vals[2*(l*sqrtR + sqrtR)] = field->Random();

    cout<<"original values:"<<endl;
    for (int i=0; i<2*(l*sqrtR + sqrtR) + 1; i++){
        cout<<(long) vals[i].elem<<endl;
    }
    return vals;
}

template<class FieldType>
vector<vector<FieldType>> Client<FieldType>::createShares(vector<FieldType> & vals){

    vector<vector<FieldType>> sharesElements(numServers);

    vector<FieldType> x1(numServers),y1(numServers);

    // prepare the shares for the input
    for (int k = 0; k < vals.size(); k++)
    {

        // the value of a_0 is the input.
        x1[0] = vals[k];

        // generate random degree-T polynomial
        for(int i = 1; i < T+1; i++)
        {
            // A random field element, uniform distribution
            x1[i] = field->Random();
        }

        matrix_vand.MatrixMult(x1, y1, T+1); // eval poly at alpha-positions predefined to be alpha_i = i

        // prepare shares to be sent
        for(int i=0; i < numServers; i++)
        {
            //cout << "y1[ " <<i<< "]" <<y1[i] << endl;
            sharesElements[i].push_back(y1[i]);

        }
    }

    return sharesElements;
}

template<class FieldType>
void Client<FieldType>::writeServersFiles(vector<vector<FieldType>> & shares, int clientID){

    ofstream outputFile;
    int size = 2*(l*sqrtR + sqrtR) + 1;

    for (int i=0; i<numServers; i++){

        if (field->getElementSizeInBytes() == 8) {
            long *serverShares = (long *) shares[i].data();
            outputFile.open("server" + to_string(i) + "ForClient" + to_string(clientID) + "inputs.txt");

            for (int j = 0; j < size; j++) {
                outputFile << serverShares[j] << endl;
            }
        }
        if (field->getElementSizeInBytes() == 4) {
            int *serverShares = (int *) shares[i].data();
            outputFile.open("server" + to_string(i) + "ForClient" + to_string(clientID) + "inputs.txt");

            for (int j = 0; j < size; j++) {
                outputFile << serverShares[j] << endl;
            }
        }
        outputFile.close();
    }

}

template<class FieldType>
void Client<FieldType>::readServerFile(string fileName, vector<FieldType> & msg, vector<FieldType> & unitVector, FieldType * e){

    ifstream inputFile;
    inputFile.open(fileName);

    int msgSize = 2*l*sqrtR + sqrtR;
    msg.resize(msgSize);

    int unitSize = sqrtR;
    unitVector.resize(unitSize);

    long input;
    for (int j=0; j<msgSize; j++) {
        inputFile >> input;
        msg[j] = input;
    }

    for (int j=0; j<unitSize; j++) {
        inputFile >> input;
        unitVector[j] = input;
    }

    inputFile >> input;
    *e = input;

    inputFile.close();

}

template<class FieldType>
void Client<FieldType>::checkServerFiles(int clientID){


    ifstream inputFile;
    int size = 2*(l*sqrtR + sqrtR) + 1;
    vector<vector<FieldType>> shares(numServers, vector<FieldType>(size));

    long input;
    for (int i=0; i<numServers; i++) {

        inputFile.open("server" + to_string(i) + "ForClient" + to_string(clientID) + "inputs.txt");

        for (int j = 0; j < size; j++) {
            inputFile >> input;
            shares[i][j] = input;
        }

        inputFile.close();
    }

    vector<FieldType> x(numServers);
    vector<FieldType> y_for_interpolate(numServers);
    HIM<FieldType> matrix_for_interpolate;
    matrix_for_interpolate.allocate(1,numServers, field);
    vector<FieldType> alpha(numServers), beta(1); // N distinct non-zero field elements
    // N distinct non-zero field elements
    for(int i=0; i<numServers; i++)
    {
        alpha[i]=field->GetElement(i+1);
    }
    beta[0] = field->GetElement(0); // zero of the field
    matrix_for_interpolate.InitHIMByVectors(alpha, beta);

    vector<FieldType> secrets(size);

    for(int k=0; k<size; k++){

        //get the set of shares for each element
        for(int i=0; i < numServers; i++) {

            x[i] = shares[i][k];
        }


        matrix_for_interpolate.MatrixMult(x, y_for_interpolate);
        secrets[k] = y_for_interpolate[0];

    }

    cout<<"opened values:"<<endl;
    for (int i=0; i<size; i++){
        cout<<(long) secrets[i].elem<<endl;
    }

}

template<class FieldType>
void Client<FieldType>::extractMessages(vector<FieldType> & messages, vector<int> & counters, int numMsgs){

    int totalNumber = 0;
    for (int i=0; i<numMsgs; i++){
        totalNumber += counters[i];
        for (int j=0; j<l;j++) {
            calcPairMessages(messages[2 * i * l + j], messages[(2 * i + 1) * l + j], counters[i]);
        }
    }

    cout<<"total number = "<<totalNumber<<endl;
    cout<<"numClients = "<<numClients<<endl;
    if (totalNumber > numClients){
        cout<<"CHEATING!!!"<<endl;
    }
}

template<class FieldType>
void Client<FieldType>::calcPairMessages(FieldType & a, FieldType & b, int counter){

    //If there is no element in this index, check that both values are zero.
    if (counter == 0){
        if (a != *(field->GetZero()) || b != *(field->GetZero())){
            cout<<"CHEATING!!!"<<endl;
        }
    //If there is one element in this index, check that x = x^2.
    } else if (counter == 1){
        FieldType temp = a*a;
        if (b == temp){
            b = *(field->GetZero());
        } else {
            cout<<"CHEATING!!!"<<endl;
        }
    //If there are two elements in this index, calculate them
    } else if (counter == 2){

        FieldType eight = field->GetElement(8);
        FieldType four = field->GetElement(4);
        FieldType two = field->GetElement(2);

        FieldType insideSqrt = eight*b - a*a*four; //8b-4a^2

        //The algorithm for checking the squrae root of a value is as follows:
        //We know that 2^31 and 2^61 are both divisible by 4 (the results are 2^29 and 2^59 respectively). So 2^31-1=3 mod 4 and 2^61-1=3 mod 4.
        //So if we have b=x^2 (over Mersenne61) then we can compute x by b^{2^59}.
        //To do this, we can make about 58 field multiplications:
            //Set b_1 = b, then
            //For i=2...59:
                //compute b_i = (b_{i-1})^2.
            //So x1=b_59 and x2=-b_59 = 2^61-1-b_59
            //Check that x1^2 = b, if it does then output it, otherwise, it means that a cheat is detected.
//        FieldType root = sqrt;
//        for (int i=2; i<=60; i++){
//            root *= root;
//        }
//        FieldType check = root*root;
//
//        if (check != sqrt){
//            cout<<"CHEATING!!!"<<endl;
//            return;
//        }
        FieldType root = insideSqrt.sqrt();

        //calculate the final messages
        FieldType val = two*a;
        //put the first message in b
        a = (val + root) / four;

        //put the second message in a
        b = (val - root) / four;

    } else {
        a = *(field->GetZero());
        b = *(field->GetZero());
    }
}


template<class FieldType>
void Client<FieldType>::checkExtractMsgs() {
    int numMsgs = sqrtR;
    vector<FieldType> messages(2*l*numMsgs, *field->GetZero());
    vector<int> counters(numMsgs);

    counters[0] = 0;
    auto first = field->Random();

    cout<<"first element = "<<first<<endl;
    for (int i=0; i<l; i++) {
        messages[2*l + i] = first;

        messages[3*l + i] = first * first;
    }
    counters[1] = 1;

    auto second = field->Random();
    cout<<"second element = "<<second<<endl;
    for (int i=0; i<l; i++) {
        messages[4*l + i] = first + second;
        messages[5*l + i] = first * first + second * second;
    }
    counters[2] = 2;

    for (int i=0; i<numMsgs; i++) {
        for (int j = 0; j < l; j++) {
            cout<<messages[2*l*i + j]<<" ";
        }
        for (int j = 0; j < l; j++) {
            cout<<messages[(2*i+1)*l + j]<<" ";
        }
        cout<<endl;
    }

    extractMessages(messages, counters, numMsgs);

    for (int i=0; i<numMsgs; i++) {
        for (int j = 0; j < l; j++) {
            cout<<messages[2*l*i + j]<<" ";
        }
        for (int j = 0; j < l; j++) {
            cout<<messages[(2*i+1)*l + j]<<" ";
        }
        cout<<endl;
    }

}

#endif //MPCANONYMOUSBLOGGINGCLIENT_CLIENT_H
