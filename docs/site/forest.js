/* forest.js — volumetric forest hero.
   One full-screen fragment shader, no dependencies.

   Trees are vertical, so a trunk is a circle in the XZ plane. Both the
   primary intersection and the sun-shadow probe collapse to a 2-D DDA walk
   through a jittered grid — which is what makes honest volumetric light
   shafts affordable on an integrated GPU.

   Lighting is kept in three separate magnitudes, because a forest does:
     direct — warm sun, hard-shadowed by trunks, gated by canopy
     fill   — GREEN. Skylight arrives through leaves, which transmit green.
     aerial — haze that goes far brighter than white toward the sun. That
              blow-out against near-black trunks is what reads as sunlight. */
(function () {
  "use strict";

  var VERT = "attribute vec2 aPos;void main(){gl_Position=vec4(aPos,0.0,1.0);}";

  var FRAG = [
    "precision highp float;",
    "uniform vec2  uRes;",
    "uniform float uTime;",
    "uniform float uWalk;",
    "uniform vec2  uPar;",
    "uniform float uSteps;",

    "#define CELL 2.15",
    "#define PI 3.14159265",

    "float h21(vec2 p){vec3 p3=fract(vec3(p.xyx)*0.1031);p3+=dot(p3,p3.yzx+33.33);return fract((p3.x+p3.y)*p3.z);}",
    "vec2 h22(vec2 p){vec3 p3=fract(vec3(p.xyx)*vec3(0.1031,0.1030,0.0973));p3+=dot(p3,p3.yzx+33.33);return fract((p3.xx+p3.yz)*p3.zy);}",
    "float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);",
    " return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y);}",
    "float fbm2(vec2 p){return 0.60*vnoise(p)+0.30*vnoise(p*2.07);}",
    "float fbm3(vec2 p){return 0.52*vnoise(p)+0.27*vnoise(p*2.07)+0.14*vnoise(p*4.11);}",
    "float fbm(vec2 p){float a=0.5,s=0.0;for(int i=0;i<4;i++){s+=a*vnoise(p);p*=2.03;a*=0.5;}return s;}",

    /* ---------------- the trunk field ---------------- */
    "vec3 treeInfo(vec2 c){",
    " vec2 j=h22(c);",
    " float r=h21(c+19.7);",
    " float rad=0.065+0.30*r*r*r;",
    " if(h21(c+3.3)>0.90) rad=0.30+0.24*h21(c+5.1);",
    " if(h21(c+61.7)<0.22) rad=0.0;",
    " return vec3(0.30+0.40*j.x,0.30+0.40*j.y,rad);}",
    "float treeHue(vec2 c){return h21(c+91.3);}",
    "float treeTop(vec2 c){return 14.5+10.0*h21(c+41.9);}",

    "float trunkTrace(vec3 ro,vec3 rd,float tmax,out vec3 nrm,out vec2 cellOut,out float radOut){",
    " nrm=vec3(0.0);cellOut=vec2(0.0);radOut=0.0;",
    " vec2 p=ro.xz,d=rd.xz;float L=length(d);",
    " if(L<1e-4) return -1.0;",
    " d/=L; float tm=tmax*L;",
    " vec2 cell=floor(p/CELL);",
    " vec2 stp=sign(d);",
    " vec2 dl=abs(vec2(CELL)/max(abs(d),vec2(1e-5)));",
    " vec2 tnx=((cell+max(stp,0.0))*CELL-p)/(d+vec2(1e-9));",
    " tnx=mix(tnx,vec2(1e9),step(abs(d),vec2(1e-5)));",
    " for(int i=0;i<24;i++){",
    "   vec3 ti=treeInfo(cell);",
    "   if(ti.z>0.0){",
    "     vec2 oc=p-(cell+ti.xy)*CELL;",
    "     float b=dot(oc,d);",
    "     float disc=b*b-(dot(oc,oc)-ti.z*ti.z);",
    "     if(disc>0.0){",
    "       float t=-b-sqrt(disc);",
    "       if(t>0.002&&t<tm){",
    "         float t3=t/L;",
    "         if(ro.y+rd.y*t3<treeTop(cell)){",
    "           vec2 hp=oc+t*d;",
    "           nrm=normalize(vec3(hp.x,0.0,hp.y));",
    "           cellOut=cell; radOut=ti.z;",
    "           return t3;}",
    "       }",
    "     }",
    "   }",
    "   if(tnx.x<tnx.y){ if(tnx.x>tm) break; cell.x+=stp.x; tnx.x+=dl.x; }",
    "   else           { if(tnx.y>tm) break; cell.y+=stp.y; tnx.y+=dl.y; }",
    " }",
    " return -1.0;}",

    /* occlusion-only probe: 8 cells is ~17 m, past which shafts are haze anyway */
    "float trunkBlocked(vec3 ro,vec3 rd,float tmax){",
    " vec2 p=ro.xz,d=rd.xz;float L=length(d);",
    " if(L<1e-4) return 0.0;",
    " d/=L; float tm=tmax*L;",
    " vec2 cell=floor(p/CELL);",
    " vec2 stp=sign(d);",
    " vec2 dl=abs(vec2(CELL)/max(abs(d),vec2(1e-5)));",
    " vec2 tnx=((cell+max(stp,0.0))*CELL-p)/(d+vec2(1e-9));",
    " tnx=mix(tnx,vec2(1e9),step(abs(d),vec2(1e-5)));",
    " for(int i=0;i<8;i++){",
    "   vec3 ti=treeInfo(cell);",
    "   if(ti.z>0.0){",
    "     vec2 oc=p-(cell+ti.xy)*CELL;",
    "     float b=dot(oc,d);",
    "     float disc=b*b-(dot(oc,oc)-ti.z*ti.z);",
    "     if(disc>0.0){ float t=-b-sqrt(disc);",
    "       if(t>0.002&&t<tm&&ro.y+rd.y*(t/L)<treeTop(cell)) return 1.0; }",
    "   }",
    "   if(tnx.x<tnx.y){ if(tnx.x>tm) break; cell.x+=stp.x; tnx.x+=dl.x; }",
    "   else           { if(tnx.y>tm) break; cell.y+=stp.y; tnx.y+=dl.y; }",
    " }",
    " return 0.0;}",

    /* ---------------- undergrowth as a height field ----------------
       Domain-warped, or the marched fronds alias into corduroy ridges. */
    "float groundH(vec2 p){",
    " float w=vnoise(p*0.62);",
    " vec2 q=p+vec2(w,0.75-w)*1.05;",
    " float clump=smoothstep(0.26,0.84,fbm2(q*1.55));",
    " float frond=0.40+0.60*vnoise(q*5.6+vec2(w*3.0,w*1.5));",
    " return 0.10*w+0.46*clump*frond;}",

    "float traceGround(vec3 ro,vec3 rd,float tmax,float jit){",
    " float t=max((0.72-ro.y)/rd.y,0.04);",
    " t+=jit*0.10;",
    " float pt=t;",
    " for(int i=0;i<15;i++){",
    "   vec3 p=ro+rd*t;",
    "   if(p.y<groundH(p.xz)) return mix(pt,t,0.5);",
    "   pt=t; t+=max(0.11,t*0.085);",
    "   if(t>tmax) break;",
    " }",
    " return -1.0;}",

    "vec3 groundN(vec2 p){",
    " vec2 e=vec2(0.055,0.0);",
    " float h=groundH(p);",
    " return normalize(vec3(h-groundH(p+e.xy),e.x,h-groundH(p+e.yx)));}",

    "float canopy(vec3 p,float t){",
    " vec2 q=p.xz*0.33+vec2(t*0.021,t*0.012);",
    " float c=fbm3(q*1.55);",
    " float tw=0.86+0.14*vnoise(p.xz*2.4+vec2(t*0.35,t*0.20));",
    " return smoothstep(0.16,0.62,c)*tw;}",

    "float hg(float c,float g){float g2=g*g;return (1.0-g2)/(4.0*PI*pow(max(1.0+g2-2.0*g*c,1e-4),1.5));}",

    "void main(){",
    " vec2 uv=(gl_FragCoord.xy*2.0-uRes)/uRes.y;",
    " float t=uTime;",
    " float dith=fract(h21(gl_FragCoord.xy)+0.6180339887*floor(t*60.0));",

    " vec3 ro=vec3(0.10*sin(t*0.13)+uPar.x*0.42, 1.62+0.030*sin(t*0.31), uWalk);",
    " float yaw=0.035*sin(t*0.089)+uPar.x*0.055;",
    " float pit=-0.058+0.014*sin(t*0.11)-uPar.y*0.045;",
    " vec3 rd=normalize(vec3(uv.x,uv.y,1.35));",
    " rd=normalize(vec3(rd.x*cos(yaw)+rd.z*sin(yaw), rd.y, rd.z*cos(yaw)-rd.x*sin(yaw)));",
    " rd=normalize(vec3(rd.x, rd.y*cos(pit)+rd.z*sin(pit), rd.z*cos(pit)-rd.y*sin(pit)));",

    " vec3 SUN=normalize(vec3(0.430,0.108,1.0));",
    " float sd=max(dot(rd,SUN),0.0);",

    /* ---- illumination magnitudes (NOT albedos) ---- */
    " vec3 sunCol =vec3(1.00,0.815,0.470);",
    " vec3 fillCol=vec3(0.175,0.495,0.235);",
    " vec3 mistCol=vec3(0.105,0.243,0.140);",

    /* ---- primary hit ---- */
    " float FAR=48.0;",
    " vec3 nrm; vec2 cell; float rad;",
    " float tT=trunkTrace(ro,rd,FAR,nrm,cell,rad);",
    " float tFlat=(rd.y<-2e-3)?(-ro.y/rd.y):-1.0;",
    " float tG=-1.0;",
    " bool detailed=false;",
    " if(tFlat>0.0){",
    "   if(tFlat<20.0){ tG=traceGround(ro,rd,22.0,dith); detailed=true; }",
    "   if(tG<0.0){ tG=(tFlat<FAR)?tFlat:-1.0; detailed=false; }",
    " }",
    " float tHit=FAR; int kind=0;",
    " if(tT>0.0){tHit=tT;kind=1;}",
    " if(tG>0.0&&tG<tHit){tHit=tG;kind=2;}",

    " float up=smoothstep(-0.02,0.34,rd.y);",
    " vec3 col;",

    /* ---- background: canopy ceiling, blown-out haze toward the sun ---- */
    " {",
    "  float leaf=0.0;",
    "  if(rd.y>0.02){ leaf=fbm(((ro+rd*((7.6-ro.y)/rd.y)).xz)*0.42+vec2(t*0.016,t*0.009)); }",
    "  float gap=smoothstep(0.46,0.80,leaf);",
    "  vec3 sky=mix(vec3(0.014,0.040,0.026),vec3(0.26,0.52,0.16),gap);",
    "  sky=mix(mistCol*1.30,sky,smoothstep(0.0,0.22,rd.y));",
    "  col=sky;",
    " }",

    /* ---- trunks ---- */
    " if(kind==1){",
    "   vec3 P=ro+rd*tHit;",
    "   float ang=atan(nrm.x,nrm.z);",
    "   float hue=treeHue(cell);",
    "   float bark=fbm(vec2(ang*3.2/max(rad,0.06),P.y*0.55))*0.85",
    "             +fbm2(vec2(ang*14.0,P.y*4.6))*0.40",
    "             +fbm2(vec2(ang*31.0,P.y*11.0))*0.17;",
    "   float moss=smoothstep(0.40,0.84,fbm2(vec2(ang*2.0,P.y*0.38+cell.x*0.7)))",
    "             *smoothstep(6.0,0.2,P.y);",
    "   vec3 alb=mix(mix(vec3(0.052,0.042,0.034),vec3(0.115,0.094,0.068),hue),",
    "                mix(vec3(0.235,0.198,0.145),vec3(0.300,0.250,0.176),hue), bark);",
    "   alb=mix(alb,vec3(0.118,0.268,0.096),moss*0.82);",
    "   float lit=max(dot(nrm,SUN),0.0);",
    "   float vis=(1.0-trunkBlocked(P+nrm*0.02,SUN,26.0))*canopy(P,t);",
    "   float fres=pow(1.0-abs(dot(nrm,rd)),2.2);",
    "   vec3 shade=fillCol*(0.42+0.58*up)*1.62+vec3(0.028,0.052,0.075)*0.85;",
    "   col =alb*(shade + sunCol*lit*vis*2.95);",
    "   col+=sunCol*fres*vis*(0.24+0.76*lit)*1.85;",
    "   col+=fillCol*fres*0.36;",
    " }",

    /* ---- undergrowth + floor ---- */
    " else if(kind==2){",
    "   vec3 P=ro+rd*tHit;",
    "   vec3 N=detailed?groundN(P.xz):vec3(0.0,1.0,0.0);",
    "   float h=detailed?groundH(P.xz):0.06;",
    "   float fern=detailed?smoothstep(0.13,0.40,h):smoothstep(0.28,0.60,fbm2(P.xz*1.55));",
    "   fern=mix(0.45,fern,0.35+0.65*smoothstep(24.0,7.0,tHit));",
    "   float lod=1.0-0.52*smoothstep(4.0,23.0,tHit);",
    "   float g=fbm2(P.xz*1.35);",
    "   float litter=fbm2(P.xz*6.5*lod);",
    "   vec3 soil=mix(vec3(0.105,0.092,0.058),vec3(0.190,0.152,0.088),litter);",
    "   vec3 leafG=mix(vec3(0.112,0.198,0.076),vec3(0.190,0.310,0.108),g);",
    "   vec3 alb=mix(soil,leafG,fern);",
    "   float vis=(1.0-trunkBlocked(P+N*0.05,SUN,30.0))*canopy(P,t);",
    "   float lit=max(dot(N,SUN),0.0);",
    "   float trans=pow(max(dot(rd,-SUN),0.0),2.5)*fern;",
    "   col =alb*(fillCol*(0.46+0.54*up)*1.78+vec3(0.026,0.048,0.070)*0.80 + sunCol*lit*vis*3.25);",
    "   col+=sunCol*alb*trans*vis*1.30;",
    " }",

    /* ---- volumetric shafts ---- */
    " float march=min(tHit,32.0);",
    " float ns=uSteps;",
    " float dt=march/ns;",
    " float shaft=0.0;",
    " for(int i=0;i<36;i++){",
    "   if(float(i)>=ns) break;",
    "   float s=(float(i)+dith)*dt;",
    "   vec3 P=ro+rd*s;",
    "   shaft+=(1.0-trunkBlocked(P,SUN,16.0))*canopy(P,t)*exp(-max(P.y,0.0)*0.150);",
    " }",
    " shaft*=dt;",
    " col+=sunCol*shaft*hg(dot(rd,SUN),0.76)*0.345;",

    /* ---- aerial perspective: blows past white toward the sun ---- */
    " float fogA=1.0-exp(-tHit*0.0268);",
    " vec3 fogCol=mistCol*0.94+sunCol*pow(sd,8.0)*2.15;",
    " col=mix(col,fogCol,fogA*0.80);",

    /* ---- sun core and bloom, occluded by whatever is in front ---- */
    " float clear=(kind==0)?1.0:0.0;",
    " col+=sunCol*pow(sd,26.0)*1.35*mix(0.30,1.0,clear)*fogA;",
    " col+=sunCol*pow(sd,340.0)*5.5*clear;",

    /* ---- pollen ---- */
    " float motes=0.0;",
    " vec2 mp=uv*3.0;",
    " for(int i=0;i<3;i++){",
    "   float fi=float(i);",
    "   vec2 q=mp*(1.0+fi*0.62)+vec2(t*(0.028+0.020*fi),-t*(0.042+0.028*fi));",
    "   vec2 gi=floor(q), gf=fract(q)-0.5;",
    "   vec2 o=h22(gi+fi*17.0)-0.5;",
    "   motes+=smoothstep(0.046,0.0,length(gf-o*0.72))*step(0.94,h21(gi+fi*3.7))",
    "        *(0.55+0.45*sin(t*1.6+h21(gi)*40.0));",
    " }",
    " col+=sunCol*motes*0.60*smoothstep(0.08,0.58,sd)*(1.0-fogA*0.35);",

    /* ---- grade ---- */
    " col=max(col,0.0);",
    " col*=1.04;",
    " col=col/(1.0+col*0.48);",
    " col=pow(col,vec3(1.00,0.895,1.06));",
    " col*=1.0-0.30*dot(uv*vec2(0.50,0.64),uv*vec2(0.50,0.64));",
    " col+=(h21(gl_FragCoord.xy*1.7+t)-0.5)*0.012;",
    " gl_FragColor=vec4(col,1.0);",
    "}"
  ].join("\n");

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("[forest] shader: " + gl.getShaderInfoLog(s)); return null;
    }
    return s;
  }

  window.initForest = function (canvas, opts) {
    opts = opts || {};
    var gl = canvas.getContext("webgl", {
      antialias: false, alpha: false, depth: false,
      powerPreference: "low-power", preserveDrawingBuffer: false
    });
    if (!gl) return null;

    var vs = compile(gl, gl.VERTEX_SHADER, VERT);
    var fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
    if (!vs || !fs) return null;
    var prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("[forest] link: " + gl.getProgramInfoLog(prog)); return null;
    }
    gl.useProgram(prog);

    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    var loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    var U = {
      res:   gl.getUniformLocation(prog, "uRes"),
      time:  gl.getUniformLocation(prog, "uTime"),
      walk:  gl.getUniformLocation(prog, "uWalk"),
      par:   gl.getUniformLocation(prog, "uPar"),
      steps: gl.getUniformLocation(prog, "uSteps")
    };

    var scale = opts.scale || 0.85, W = 0, H = 0;
    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 1.6);
      var w = Math.max(1, Math.round(canvas.clientWidth  * dpr * scale));
      var h = Math.max(1, Math.round(canvas.clientHeight * dpr * scale));
      if (w === W && h === H) return;
      W = w; H = h; canvas.width = w; canvas.height = h;
      gl.viewport(0, 0, w, h);
    }

    var api = { walk: 0, par: [0, 0], steps: opts.steps || 16, gl: gl };
    api.render = function (tSec) {
      resize();
      gl.uniform2f(U.res, W, H);
      gl.uniform1f(U.time, tSec);
      gl.uniform1f(U.walk, api.walk);
      gl.uniform2f(U.par, api.par[0], api.par[1]);
      gl.uniform1f(U.steps, api.steps);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    api.setScale = function (s) { scale = s; W = H = 0; };
    return api;
  };
})();
