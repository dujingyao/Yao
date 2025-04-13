#include<bits/stdc++.h>
using namespace std;

const int N=5e5+10;

int main(){
    vector<double> a;
    double s,m;
    int n;
    scanf("%d%lf",&n,&s);
    for(int i=0;i<n;i++){
        scanf("%lf",&m);
        a.push_back(m);
    }
    sort(a.begin(),a.end());
    double average;
    average=s/(1.0*n);
    double allaverage=average;
    int cnt=0;
    double sumfixed=0.0;
    for(int i=0;i<a.size();i++){
        if(a[i]<average){
            sumfixed+=a[i];
            cnt++;//当前有多少个被加入进去了
            s-=a[i];
            average=s/(n-cnt);
        }else{
            //剩余元素>=average停止处理
            break;
        }
    }
    double ans=0.0;
    for(int i=0;i<cnt;i++){
        //当删除掉小于初始平均值的元素后
        //剩下的平均值只会比初始的平均值小
        //因此所有元素均大于这个数
        ans=ans+(a[i]-allaverage)*(a[i]-allaverage);
    }
    if(n>cnt){
        double diff=(average-allaverage);
        ans+=(n-cnt)*diff*diff;
    }
    printf("%.4lf",sqrt(ans/n));
    return 0;
}