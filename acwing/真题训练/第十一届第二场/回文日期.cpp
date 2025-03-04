#include<bits/stdc++.h>
using namespace std;
//回文日期判断
//返回1则是回文日期但不是特定格式
//返回2是ABABBABA型回文日期
int f(string s){
    if(s[0]==s[2]&&s[0]==s[5]&&s[0]==s[7]&&
       s[1]==s[3]&&s[1]==s[4]&&s[1]==s[6]&&s[0]!=s[1]){
        return 2;
    }
    for(int i=0;i<=3;i++){
        if(s[i]!=s[7-i]) return 0;
    }
    return 1;
}

string turn(int day){
    int i=10000000;
    string ch;
    while(i){
        ch+=(day/i%10+'0');
        i/=10;
    }
    return ch;
}

bool check(int num){
    int year=num/10000;
    int month=num%10000/100;
    int day=num%100;
    if(month>12||month<1) return false;
    if(month==2){
        if(year%4==0&&year%100!=0||year%400==0){//闰年29
            if(day>29) return false;
        }
        else{
            if(day>28) return false;
        }
    }
    else if(month==1||month==3||month==5||month==7||month==8||month==10||month==12){
        if(day>31) return false;
    }
    else{
        if(day>30) return false;
    }
    return true;
}

int main(){
    int num,level=0,alevel=0;
    cin>>num;
    for(int i=num+1;i<=99999999;i++){
        if(check(i)){
            string ch=turn(i);
            if(f(ch)==1&&level==0||f(ch)==2&&level==0) level=i;
            if(f(ch)==2&&alevel==0) alevel=i;
        }
        if(level!=0&&alevel!=0) break;
    }
    cout<<level<<endl;
    cout<<alevel;
    return 0;
}