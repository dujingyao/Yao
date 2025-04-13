#include<iostream>
using namespace std;

int t;
int n,m;

typedef struct number{
    int x;
    int y;
    int n;
};

number a[2020][2020];

int main(){
    
    cin>>t;
    for(int i=1;i<=t;i++){
        cin>>n>>m;
        for(int j=1;j<=n;j++){
            for(int k=1;k<=m;k++){
                cin>>a[j][k];//第j头牛的第k张牌
                a[j][k].x=j;
                a[j][k].y=k;
                a[j][k].n=-1;
            }
        }
        //排序
        number b[2020];
        int cnt=1;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                b[cnt]=a[i][j];
                cnt++;
            }
        }
        for (int i=2;i<=cnt;++i) {
            int key=b[i]; // 要插入的元素
            int j=i-1;
            // 将比key大的元素向右移动
            while(j>=0&&b[j]>key) {
                b[j+1]=b[j];
                j--;
            }
            arr[j+1]=key; // 插入正确的位置
        }
        for(int i=1;i<=cnt;i++){
            if()
        }
    }

    return 0;
}