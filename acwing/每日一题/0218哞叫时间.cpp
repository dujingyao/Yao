#include<iostream>
#include<string>
#include<vector>
using namespace std;

string s;

vector<string> ch;

int n,m;

int main(){
    
    string a;
    cin>>n>>m;
    cin>>s;
    for(char c1='a';c1<='z';c1++){
        for(char c2='a';c2<='z';c2++){
            if(c1==c2) continue;
            int cnt=0,flag=0;
            string t=s;
            //原来存在
            for(int i=0;i<n;i++){
                if(t[i]==c1&&t[i+1]==c2&&t[i+2]==c2){
                    t[i]=t[i+1]=t[i+2]='#';
                    cnt+=1;
                }
            }
            //需要更改
            for(int i=0;i<n;i++){
                if(t[i]=='#'||t[i+1]=='#'||t[i+2]=='#') continue;
                if(t[i]!=c1&&t[i+1]==c2&&t[i+2]==c2||
                t[i]==c1&&t[i+1]!=c2&&t[i+2]==c2||
                t[i]==c1&&t[i+1]==c2&&t[i+2]!=c2){
                    t[i]=t[i+1]=t[i+2]='#';
                    flag=1;
                }
            }
            if(cnt+flag>=m){
                string b;
                b+=c1;
                b+=c2;
                b+=c2;
                ch.push_back(b);
            }
        }
    }
    cout<<ch.size()<<endl;
    for(int i=0;i<ch.size();i++){
        cout<<ch[i]<<endl;
    }
    return 0;
}