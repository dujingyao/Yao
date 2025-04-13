#include<bits/stdc++.h>
using namespace std;

bool check(long long int a){
    char st[8];
    for(int i=0;i<8;i++){
        st[i]=a%10;
        a/=10;
    }
    for(int i=0;i<4;i++){
        if(st[i]!=st[7-i]) return false;
    }
    return true;
}

long long int up(long long int a){
    int year,month,day;
    day=a%100;
    a/=100;
    month=a%100;
    a/=100;
    year=a;
    if(month==1||month==3||month==5||month==7||month==8||month==10||month==12){//31天
        day++;
        if(day==32){
            day=1;
            month++;
        }
        if(month==13){
            month=1;
            year++;
        }
    }
    else if(month==2){
        day++;
        if(year%4==0&&year%100!=0||year%400==0){//闰年，29
            if(day==30){
                day=1;
                month++;
            }
        }
        else{
            if(day==29){
                day=1;
                month++;
            }
        }
    }
    else{
        day++;
        if(day==31){
            day=1;
            month++;
        }
    }
    return year*10000+month*100+day;
}

int main(){
    
    long long int day1,day2;
    cin>>day1>>day2;
    int res=0;
    while(day1<=day2){
        if(check(day1)) res++;
        day1=up(day1);
    }
    cout<<res<<endl;

    return 0;
}