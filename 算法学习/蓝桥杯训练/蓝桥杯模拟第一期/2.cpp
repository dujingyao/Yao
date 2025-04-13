#include<iostream>
#include<cmath>
using namespace std;
int main(){
    int a=2024,b=0;
    while(a>1){
        a=(int)sqrt(a);
        b++;
    }
    cout<<b<<endl;
    return 0;
}