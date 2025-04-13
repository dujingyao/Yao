#include<bits/stdc++.h>
using namespace std;

int main(){
    
    int w,m,n;
    cin>>w>>m>>n;
    //行数应该向上取整
    int row1,row2;//行
    int col1,col2;//列
    //算出是第几行
    row1=(m+w-1)/w,row2=(n+w-1)/w;
    //算出是第几列
    if(row1%2==1){
        col1=m%w;
        if(col1==0) col1=w;
    }else{
        col1=w+1-m%w;
        if(col1==w+1) col1=1;
    }
    if(row2%2==1){
        col2=n%w;
        if(col2==0) col2=w;
    }else{
        col2=w+1-n%w;
        if(col2==w+1) col2=1;
    }
    cout<<abs(row2-row1)+abs(col1-col2)<<endl;

    return 0;
}