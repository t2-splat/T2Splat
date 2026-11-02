// Greedy, orientable surface construction. No vertex duplication or movement.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>
namespace py = pybind11;
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
struct ParityDSU {
    std::vector<int32_t> parent, size;
    std::vector<uint8_t> parity;
    explicit ParityDSU(size_t n):parent(n,-1),size(n,1),parity(n,0){}
    std::pair<int32_t,uint8_t> find(int32_t x) {
        if(parent[x]==x) return {x,0};
        auto q=find(parent[x]);parity[x]^=q.second;parent[x]=q.first;
        return {parent[x],parity[x]};
    }
    void join(int32_t a,int32_t b,uint8_t required) {
        auto x=find(a), y=find(b);
        if(x.first==y.first) return;
        uint8_t p=x.second^y.second^required;
        if(size[x.first]<size[y.first]) std::swap(x,y);
        parent[y.first]=x.first;parity[y.first]=p;size[x.first]+=size[y.first];
    }
};
py::tuple select_surface(Array<int32_t> faces, Array<int32_t> face_edges,
                         Array<int64_t> order, Array<int64_t> offsets,
                         Array<int32_t> incident, int64_t vertex_count) {
    if(faces.ndim()!=2 || faces.shape(1)!=3 || face_edges.ndim()!=2 ||
       face_edges.shape(0)!=faces.shape(0) || face_edges.shape(1)!=3 ||
       order.ndim()!=1 || offsets.ndim()!=1 || offsets.size()<1 || incident.ndim()!=1 || vertex_count<0)
        throw std::invalid_argument("Invalid selector array shapes");
    const int64_t F=faces.shape(0), E=offsets.size()-1;
    const auto* f=faces.data();const auto* fe=face_edges.data();const auto* ord=order.data();
    const auto* off=offsets.data();const auto* inc=incident.data();
    for(int64_t i=0;i<3*F;i++) if(f[i]<0 || f[i]>=vertex_count || fe[i]<0 || fe[i]>=E)
        throw std::invalid_argument("Invalid face or edge index");
    if(off[0]!=0 || off[E]!=incident.size()) throw std::invalid_argument("Invalid CSR offsets");
    for(int64_t e=0;e<E;e++) if(off[e]>off[e+1]) throw std::invalid_argument("Unsorted CSR offsets");
    std::vector<int64_t> rank(F,-1);
    for(int64_t i=0;i<order.size();i++) {
        if(ord[i]<0 || ord[i]>=F || rank[ord[i]]>=0) throw std::invalid_argument("Invalid candidate order");
        rank[ord[i]]=i;
    }
    for(int64_t i=0;i<incident.size();i++) if(inc[i]<0 || inc[i]>=F)
        throw std::invalid_argument("Invalid incident face");
    py::array_t<uint8_t> keep(F), flip(F);
    auto* kp=keep.mutable_data();auto* fp=flip.mutable_data();
    std::fill(kp,kp+F,0);std::fill(fp,fp+F,0);
    int64_t accepted=0,edge_rejected=0,duplicate_rejected=0,orientation_rejected=0,fan_attempts=0,attempts=0;
    int64_t fan_remaining=0;
    {
      py::gil_scoped_release release;
      std::vector<uint8_t> counts(E,0), used(vertex_count,0), state(F,0), queued(F,0), owner_dir(E,0);
      std::vector<int32_t> owner(E,-1);
      ParityDSU dsu(F);
      std::priority_queue<int64_t,std::vector<int64_t>,std::greater<int64_t>> retry;
      int64_t cursor=0;
      while(cursor<order.size() || !retry.empty()) {
        int64_t id;
        if(!retry.empty()) {
            id=ord[retry.top()];retry.pop();queued[id]=0;
            if(state[id]>=2) continue;
        } else {
            id=ord[cursor++];if(state[id]!=0) continue;
        }
        ++attempts;
        std::array<int32_t,3> v{f[id*3],f[id*3+1],f[id*3+2]};
        std::array<int32_t,3> es{fe[id*3],fe[id*3+1],fe[id*3+2]};
        std::array<uint8_t,3> dirs{uint8_t(v[0]<v[1]),uint8_t(v[1]<v[2]),uint8_t(v[2]<v[0])};
        if(counts[es[0]]==2 || counts[es[1]]==2 || counts[es[2]]==2) {
            state[id]=3;++edge_rejected;continue;
        }
        bool duplicate=false;
        for(int j=0;j<3;j++) if(counts[es[j]]) {
            int32_t other=owner[es[j]];
            std::array<int32_t,3> u{f[3*other],f[3*other+1],f[3*other+2]}, w=v;
            std::sort(u.begin(),u.end());std::sort(w.begin(),w.end());
            if(u==w) {duplicate=true;break;}
        }
        if(duplicate) {state[id]=3;++duplicate_rejected;continue;}
        // A valid vertex link is one path or cycle. Adding the opposite
        // link edge must attach to the existing path at each used vertex.
        bool fan_bad=false;
        for(int j=0;j<3;j++) if(used[v[j]] && counts[es[j]]==0 && counts[es[(j+2)%3]]==0) fan_bad=true;
        if(fan_bad) {state[id]=1;++fan_attempts;continue;}
        std::array<int32_t,3> roots{-1,-1,-1}, others{-1,-1,-1};
        std::array<uint8_t,3> rel{0,0,0}, wanted{0,0,0};
        bool orient_bad=false;
        for(int j=0;j<3;j++) if(counts[es[j]]) {
            others[j]=owner[es[j]];auto q=dsu.find(others[j]);roots[j]=q.first;
            wanted[j]=uint8_t(dirs[j]==owner_dir[es[j]]);rel[j]=wanted[j]^q.second;
            for(int k=0;k<j;k++) if(roots[k]==roots[j] && rel[k]!=rel[j]) orient_bad=true;
        }
        if(orient_bad) {state[id]=3;++orientation_rejected;continue;}
        state[id]=2;kp[id]=1;++accepted;dsu.parent[id]=int32_t(id);
        for(int j=0;j<3;j++) if(others[j]>=0) dsu.join(int32_t(id),others[j],wanted[j]);
        for(int j=0;j<3;j++) {
            used[v[j]]=1;const int32_t e=es[j];
            if(counts[e]++==0) {
                owner[e]=int32_t(id);owner_dir[e]=dirs[j];
                // Grow one surface component along its frontier before seeding
                // another. Global independent seeds cannot later merge without
                // temporarily disconnecting a vertex link. Pending fan faces
                // become eligible only when another edge first gains a face.
                for(int64_t k=off[e];k<off[e+1];k++) {
                    const int32_t candidate=inc[k];
                    if(state[candidate]<2 && rank[candidate]>=0 && !queued[candidate]) {
                        queued[candidate]=1;retry.push(rank[candidate]);
                    }
                }
            }
        }
      }
      for(int64_t i=0;i<F;i++) {
          if(kp[i]) fp[i]=dsu.find(int32_t(i)).second;
          else if(state[i]==1) ++fan_remaining;
      }
    }
    py::dict stats;
    stats["accepted"]=accepted;stats["edge_rejected"]=edge_rejected;
    stats["duplicate_rejected"]=duplicate_rejected;stats["orientation_rejected"]=orientation_rejected;
    stats["fan_deferred_attempts"]=fan_attempts;stats["fan_rejected"]=fan_remaining;stats["attempts"]=attempts;
    return py::make_tuple(keep,flip,stats);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("select_surface", &select_surface);}
