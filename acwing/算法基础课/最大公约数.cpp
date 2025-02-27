#include<iostream>
#include<algorithm>
using namespace std;

int main(){
    
    int n;
    cin>>n;
    while(n--){
        int a,b;
        cin>>a>>b;
        while(b){
            int c=a%b;
            a=b;
            b=c;
        }
        cout<<a<<endl;
    }

    return 0;
}