#include<bits/stdc++.h>
using namespace std;
int l,r;
const int N=1010;
char ch[N];
int t;
//向内扩散
bool check(int a,int b){
    for(int i=a,j=b;i<j;i++,j--){
        if(ch[i]!=ch[j]) return false;
    }
    return true;
}
//向外扩散
bool check1(int a,int b){
    if(a==0) return true;
    if(a>strlen(ch)-(b+1)) return false;
    for(int i=a,j=b;i>=0;i--,j++){
        if(ch[i]!=ch[j]) return false;
    }
    return true;
}
int main(){
    cin>>t;
    while(t--){
        cin>>ch;
        //寻找前置
        for(int i=0;i<strlen(ch);i++){
            if(ch[i]=='l'||ch[i]=='q'||ch[i]=='b') continue;
            else{
                l=i;
                break;
            }
        }
        //寻找后置
        for(int i=strlen(ch)-1;i>=0;i--){
            if(ch[i]=='l'||ch[i]=='q'||ch[i]=='b') continue;
            else{
                r=i;
                break;
            }
        }
        if(check(l,r)&&check1(l,r)){
            cout<<"Yes"<<endl;
        }
        else cout<<"No"<<endl;
    }
    return 0;
}