#include<bits/stdc++.h>
using namespace std;

//涉及scanf的新用法

int get_s(int h,int m,int s){
    return h*3600+m*60+s;
}

int get_t(){
    string line;
    getline(cin,line);
    if(line.back()!=')') line+=" (+0)";
    int h1,m1,s1,h2,m2,s2,d;
    sscanf(line.c_str(),"%d:%d:%d %d:%d:%d (+%d)",&h1,&m1,&s1,&h2,&m2,&s2,&d);

    return get_s(h2,m2,s2)-get_s(h1,m1,s1)+d*24*3600;
}

int main(){
    
    int T;
    cin>>T;
    string line;
    getline(cin,line);//忽略第一次回车
    while(T--){
        int t=(get_t()+get_t())/2;
        int h=t/3600,m=t%3600/60,s=t%60;
        printf("%02d:%02d:%02d\n",h,m,s);
    }

    return 0;
}