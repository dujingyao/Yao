#include<iostream>
using namespace std;
//引用传递
void swap(int &a,int &b){
    int temp=a;
    a=b;
    b=temp;
}

int main(){
    //引用必须初始化
    /*
    int a=10,c=20;
    int &b=a;
    cout<<"a="<<a<<endl;
    cout<<"c="<<c<<endl;
    swap(a,c);
    cout<<"a="<<a<<endl;
    cout<<"c="<<c<<endl;
    */
    
    return 0;
}