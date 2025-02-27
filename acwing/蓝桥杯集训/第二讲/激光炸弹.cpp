#include<bits/stdc++.h>
using namespace std;

const int N=5010;

int r;
int m,cnt,n,x,y,w1;
int s[N][N];

int main(){
    
    cin>>cnt>>r;
    r=min(5001,r);
    m=n=r;
    for(int i=1;i<=cnt;i++){
        cin>>x>>y>>w1;
        x++;
        y++;
        n=max(n,x);
        m=max(m,y);
        s[x][y]+=w1;
    }
    //计算前缀和
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            s[i][j]=s[i-1][j]+s[i][j-1]-s[i-1][j-1]+s[i][j];
        }
    }
    int res = 0;

    // 枚举所有边长是R的矩形，枚举(i, j)为右下角
    for (int i = r; i <= n; i ++ )
        for (int j = r; j <= m; j ++ )
            res = max(res, s[i][j] - s[i - r][j] - s[i][j - r] + s[i - r][j - r]);

    cout<<res<<endl;
    

    return 0;
}