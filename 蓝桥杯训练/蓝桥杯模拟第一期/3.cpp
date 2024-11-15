#include<iostream>
using namespace std;
int main(){
    int a=1;
    while(a*a*a<=2024){
        a++;
    }
    cout<<a*a*a-2024<<endl;
    return 0;
}