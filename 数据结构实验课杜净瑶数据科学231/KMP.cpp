#include<iostream>
#include<string>
#include<cstring>
using namespace std;
int KMP(string S,string T,int pos,int next[]){
    int i=pos;
    int j=1;
    while(i<S.length()&&j<=T.length()){
        if(j==0||S[i]==T[i]){
            i++;
            j++;
        }
        else j=next[j];
    }
    if(j>T.length()) return i-T.length();
    else return 0;
}
void get_next(string T,int next[]){
    int i=1;
    next[1]=0;
    int j=0;
    while(i<T.length()){
        if(j==0||T[i-1]==T[j-1]){
            i++;
            j++;
            next[i]=j;
        }
        else j=next[j];
    }
}
void get_nextval(string T,int nextval[]){
    int i=1;
    nextval[1]=0;
    int j=0;
    while(i<T.length()){
        if(j==0||T[i-1]==T[j-1]){
            i++;
            j++;
            if(T[i-1]!=T[j-1]) nextval[i]=j;
            else nextval[i]=nextval[j];
        }
        else j=nextval[j];
    }
}
int main(){
    string T;
    cout<<"请输入字符串：";
    cin>>T;
    cout<<endl;
    int next[100],nextval[100];
    get_next(T,next);
    get_nextval(T,nextval);
    cout<<"next:";
    for(int i=1;i<=T.length();i++){
        cout<<next[i]<<' ';
    }
    cout<<endl;
    cout<<"nextval:";
    for(int i=1;i<=T.length();i++){
        cout<<nextval[i]<<' ';
    }
    cout<<endl;
    return 0;
}