#include<iostream>
using namespace std;
//给定一个按照升序排列的长度为n的整数数组，以及q个查询。
//对于每个查询，返回一个元素k的起始位置终止位置（位置从0开始计数。
//如果数组中不存在该元素，则返回-1 -1。
const int N=100010;
int n,m;
int q[N];
int main(){
    cin>>n>>m;   //数组的长度为n,查询m次
    for(int i=0;i<n;i++) cin>>q[i];
    while(m--){
        int x;
        cin>>x;
        int l=0,r=n-1;
        while(l<r){
            int mid=l+r>>1;
            if(q[mid]>=x) r=mid;
            else l=mid+1;
        }
        if(q[l]!=x) cout<<"-1 -1"<<endl;
        else {
            cout<<l<<' ';
            int l=0,r=n-1;
            while(l<r){
                int mid=l+r+1>>1;
                if(q[mid]<=x) l=mid;
                else r=mid-1;
            }
            cout<<l<<endl;
        }
    }
    return 0;
}