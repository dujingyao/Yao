#include<bits/stdc++.h>
using namespace std;
//计算有几位小数
//小数乘大数
int main(){
    int n;
    cin>>n;
    string ch;
    cin>>ch;
    int num=0;//num位小数
    reverse(ch.begin(),ch.end());
    for(int i=0;i<ch.size();i++){
        if(ch[i]=='.') break;
        else num++;
        if(i==ch.size()-1&&ch[i]!='.') num=0;
    }
    ch.erase(num,1);
    while(n--){
        int d=0,sum;//进位的数
        for(int i=0;i<ch.size();i++){
            // char x=ch[i];
            // ch[i]=((ch[i]-'0')*2+d)%10+'0';
            // d=(x-'0')*2/10;
            sum=(ch[i]-'0')*2+d;
            ch[i]=sum%10+'0';
            d=sum/10;
        }
        while(d){
            ch+=(d%10)+'0';
            d/=10;
        }
    }
    if(num==0){
        for(int i=ch.size();i>=0;i--) cout<<ch[i];
    }
    else{
        if(ch[num-1]>='5'){
            int d=0;
            for(int i=num;i<ch.size();i++){
                if(ch[i]=='9') ch[i]='0';
                else{
                    ch[i]+=1;
                    break;
                }
            }
            if(ch[ch.size()-1]=='0') ch+='1';
        }
    }
    for(int i=ch.size()-1;i>=num;i--){
        cout<<ch[i];
    }
    
    return 0;
}