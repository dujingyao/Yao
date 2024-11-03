#include<iostream>
#include<string>
using namespace std;
int func(string ch){
    int len=ch.size();
    int sum=0;
    for(int i=0;i<len;){
        int value=0;
        switch(ch[i]){
            case 'I':value=1;break;
            case 'V':value=5;break;
            case 'X':value=10;break;
            case 'L':value=50;break;
            case 'C':value=100;break;
            case 'D':value=500;break;
            case 'M':value=1000;break;
        }
        int value1=0;
        if(i<len-1){
            switch(ch[i+1]){
            case 'I':value1=1;break;
            case 'V':value1=5;break;
            case 'X':value1=10;break;
            case 'L':value1=50;break;
            case 'C':value1=100;break;
            case 'D':value1=500;break;
            case 'M':value1=1000;break;
            }
        }
        if(value1>value){
            sum+=value1-value;
            i+=2;
        }
        else{
            sum+=value;
            i+=1;
        }
    }
    return sum;
}

int main(){
    int n;
    scanf("%d",&n);
    while(n--){
        string ch;
        cin>>ch;
        int x=func(ch);
        cout<<x<<endl;
    }

    return 0;
}