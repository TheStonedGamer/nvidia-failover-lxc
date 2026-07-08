(function(){
 let b=document.body, t=document.getElementById("timer"), c=document.querySelector(".connection");
 let lm=0, ils=0, rc, src;

function beat(){
   lm=Date.now(); b.classList.add("online"); b.classList.remove("offline");
   if(c) c.textContent="down-tick";
  }

  function chk(){
   let now=Date.now();
   if(now-lm>2000){ b.classList.remove("online"); b.classList.add("offline"); if(c) c.textContent=rc?"disconnect":"wait"; }
   if(t){ let d=(now-ils)/1000; t.textContent=d<2?((d*1000)|0)+"ms":(d|0)+"s"; }
  }

 function ssrc(){
  if(src) src.close();
  src=new EventSource("/updates");
  src.onmessage=function(e){
   var tb=document.querySelector("tbody");
   if(tb) tb.innerHTML=e.data;
   ils=Date.now(); beat();
  };
src.onerror=function(){
    b.classList.remove("online"); b.classList.add("offline");
    clearTimeout(rc); rc=setTimeout(ssrc,500);
    src.close();
   };
 }

 ils=Date.now(); ssrc(); beat();
 setInterval(chk,1000);
})();

(function(){
  var list=document.getElementById("cfglist");
  if(!list) return;
  var dragEl=null;
  list.addEventListener("dragstart",function(e){
    dragEl=e.target.closest(".cfgrow");
    if(dragEl) dragEl.classList.add("dragging");
  });
  list.addEventListener("dragend",function(){
    if(dragEl) dragEl.classList.remove("dragging");
    dragEl=null;
  });
  list.addEventListener("dragover",function(e){
    e.preventDefault();
    if(!dragEl) return;
    var after=getAfter(list,e.clientY);
    if(after==null) list.appendChild(dragEl);
    else list.insertBefore(dragEl,after);
  });
  function getAfter(container,y){
    var els=[].slice.call(container.querySelectorAll(".cfgrow:not(.dragging)"));
    var closest={offset:-Infinity,el:null};
    els.forEach(function(c){
      var box=c.getBoundingClientRect();
      var off=y-box.top-box.height/2;
      if(off<0&&off>closest.offset){closest={offset:off,el:c};}
    });
    return closest.el;
  }
  function collect(){
    var order=[],disabled=[];
    [].slice.call(list.querySelectorAll(".cfgrow")).forEach(function(r){
      var m=r.getAttribute("data-model");
      order.push(m);
      if(!r.querySelector(".tog").checked) disabled.push(m);
    });
    return {order:order,disabled:disabled};
  }
  var st=document.getElementById("cfgstatus");
  document.getElementById("cfgsave").addEventListener("click",function(){
    st.textContent="saving…";
    fetch("/_config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collect())})
      .then(function(r){return r.json();})
      .then(function(d){ st.textContent=d.ok?("saved — "+d.active.length+" active model(s)"):("error: "+(d.error||"?")); })
      .catch(function(){ st.textContent="save failed"; });
  });
  document.getElementById("cfgreset").addEventListener("click",function(){ location.reload(); });
  var ts=document.getElementById("toolstatus");
  function post(url,ok){ fetch(url,{method:"POST"})
      .then(function(r){return r.json();})
      .then(function(d){ ts.textContent=d.ok?ok(d):("error: "+(d.error||"?")); })
      .catch(function(){ ts.textContent="request failed"; }); }
  var rc=document.getElementById("rstcool");
  if(rc) rc.addEventListener("click",function(){
    ts.textContent="clearing cooldowns…";
    post("/_reset_cooldowns",function(d){return "cooldowns cleared ("+d.cleared+" model(s) revived)";});
  });
  var rs=document.getElementById("rststats");
  if(rs) rs.addEventListener("click",function(){
    if(!confirm("Zero all counters, tokens, and learned rate limits? This cannot be undone.")) return;
    ts.textContent="resetting stats…";
    post("/_reset_stats",function(d){return "stats reset";});
  });
})();

(function(){
  var st=document.getElementById("setstatus");
  function esc(s){ return String(s).replace(/[&<>"]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c];}); }
  function say(m){ if(st) st.textContent=m; }

  function addLadderRow(model){
    var list=document.getElementById("cfglist");
    if(!list) return false;
    var have=[].slice.call(list.querySelectorAll(".cfgrow")).some(function(r){return r.getAttribute("data-model")===model;});
    if(have) return false;
    var li=document.createElement("li");
    li.className="cfgrow"; li.setAttribute("draggable","true"); li.setAttribute("data-model",model);
    li.innerHTML='<span class="grip">&#8942;&#8942;</span><input type="checkbox" class="tog" checked><span class="cfgname">'+esc(model)+'</span>';
    list.appendChild(li);
    return true;
  }

  var ICONS={};
  function renderSettings(d){
    (d.providers||[]).forEach(function(p){ ICONS[p.name]=p.icon; });
    (d.presets||[]).forEach(function(p){ if(!ICONS[p.name]) ICONS[p.name]=p.icon; });
    // Preset quick-add chips — click to prefill name + base URL.
    var pr=document.getElementById("presetrow");
    if(pr && !pr.dataset.done){
      pr.dataset.done="1";
      (d.presets||[]).forEach(function(p){
        var c=document.createElement("button"); c.type="button"; c.className="presetchip";
        c.innerHTML=p.icon+"<span>"+esc(p.name)+"</span>";
        c.addEventListener("click",function(){
          document.getElementById("pvname").value=p.name;
          document.getElementById("pvurl").value=p.base_url;
          document.getElementById("pvkey").focus();
          say("prefilled "+p.name+" — add your API key + model ids");
        });
        pr.appendChild(c);
      });
    }
    var ul=document.getElementById("provlist");
    if(ul){
      ul.innerHTML="";
      if(!d.providers.length){ ul.innerHTML='<li class="pvmeta">no providers yet — pick one above or add a custom API</li>'; }
      d.providers.forEach(function(p){
        var li=document.createElement("li");
        li.innerHTML=(p.icon||"")+'<b>'+esc(p.name)+'</b> <span class="pvmeta">'+esc(p.base_url)+' · key '+(p.key_masked?esc(p.key_masked):"none")+' · '+p.models.length+' model(s)</span>';
        var b=document.createElement("button"); b.className="rm"; b.textContent="remove";
        b.addEventListener("click",function(){ post({action:"remove_provider",name:p.name}); });
        li.appendChild(b); ul.appendChild(li);
      });
    }
    // Local tail model dropdown — populate from ollama provider models.
    var sel=document.getElementById("tailsel");
    if(sel && d.local_tail){
      sel.innerHTML="";
      var opts=d.local_tail.options||[];
      if(!opts.length){
        var o=document.createElement("option"); o.textContent="(no Ollama models)"; o.disabled=true; sel.appendChild(o);
      } else {
        opts.forEach(function(m){
          var o=document.createElement("option"); o.value=m; o.textContent=m;
          if(m===d.local_tail.current) o.selected=true;
          sel.appendChild(o);
        });
      }
    }
  }

  function load(){ fetch("/_settings").then(function(r){return r.json();}).then(renderSettings).catch(function(){}); }
  function post(payload){
    say("saving…");
    return fetch("/_settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)})
      .then(function(r){return r.json();})
      .then(function(d){ if(d.error){say("error: "+d.error);} else {say("saved"); renderSettings(d);} return d; })
      .catch(function(){ say("request failed"); });
  }

  var pa=document.getElementById("pvadd");
  if(pa) pa.addEventListener("click",function(){
    var name=document.getElementById("pvname").value.trim();
    var url=document.getElementById("pvurl").value.trim();
    var key=document.getElementById("pvkey").value;
    var models=document.getElementById("pvmodels").value;
    if(!name||!url){ say("name and base URL required"); return; }
    post({action:"add_provider",name:name,base_url:url,api_key:key,models:models}).then(function(d){
      if(d&&!d.error){
        (models.split(",")).forEach(function(m){ m=m.trim(); if(m) addLadderRow(m); });
        say("provider added — models appended to ladder");
        document.getElementById("pvname").value=""; document.getElementById("pvurl").value="";
        document.getElementById("pvkey").value=""; document.getElementById("pvmodels").value="";
      }
    });
  });

  var ts=document.getElementById("tailsave");
  if(ts) ts.addEventListener("click",function(){
    var sel=document.getElementById("tailsel");
    if(!sel) return;
    var model=sel.value||"";
    post({action:"set_local_tail",model:model}).then(function(d){
      if(d&&!d.error) say("local tail model updated to: "+model);
    });
  });

  var discData=null, discHave={};
  // Family a model id lands in, after any provider-wide prefix (e.g. Google's
  // "models/") is stripped: vendor when there's a "/" ("openai/gpt-5"->"openai"),
  // else the leading family token ("gemini-3.1-flash"->"gemini", "qwen3:30b"->
  // "qwen", "phi4-mini"->"phi"). Every provider gets meaningful groups this way.
  function firstSeg(m){ var i=m.indexOf("/"); return i>0?m.slice(0,i):null; }
  function familyKey(s){
    var sl=s.indexOf("/"); if(sl>0) return s.slice(0,sl);
    var c=s.indexOf(":"); var base=c>0?s.slice(0,c):s;
    var mt=base.match(/^[A-Za-z]+/);
    return mt?mt[0]:base;
  }
  // If every id in a provider shares one leading "seg/" prefix, strip it so
  // grouping keys off what varies (Google's ids are all under "models/").
  function groupKeyer(models){
    var segs={}, allSlash=true;
    models.forEach(function(m){ var f=firstSeg(m); if(f===null) allSlash=false; else segs[f]=1; });
    var ks=Object.keys(segs), strip=(allSlash&&ks.length===1)?(ks[0]+"/"):null;
    return function(m){ var s=(strip&&m.indexOf(strip)===0)?m.slice(strip.length):m; return familyKey(s); };
  }
  function mdlChip(m){
    var chip=document.createElement("span"); chip.className="mdlchip"+(discHave[m]?" have":"");
    chip.appendChild(document.createTextNode(m));
    if(!discHave[m]){
      var add=document.createElement("button"); add.textContent="＋";
      add.addEventListener("click",function(){ if(addLadderRow(m)){ discHave[m]=1; chip.className="mdlchip have"; add.remove(); say("added "+m+' — hit "Save order & toggles"'); } });
      chip.appendChild(add);
    }
    return chip;
  }
  function addAllBtn(list,cls,label){
    var addable=list.filter(function(m){ return !discHave[m]; });
    if(!addable.length) return null;
    var b=document.createElement("button"); b.type="button"; b.className=cls;
    b.textContent="＋ add all "+addable.length+(label||" shown");
    b.addEventListener("click",function(e){ e.preventDefault(); var n=0; addable.forEach(function(m){ if(addLadderRow(m)){ discHave[m]=1; n++; } }); say("added "+n+' model(s) — hit "Save order & toggles"'); renderDisc(); });
    return b;
  }
  function renderDisc(){
    var box=document.getElementById("discresult");
    if(!box||!discData) return;
    var pv=discData.providers||{};
    var names=Object.keys(pv);
    box.innerHTML="";
    if(!names.length){ box.appendChild(document.createTextNode("no providers configured — add one above")); return; }
    var q=(document.getElementById("discfilter")||{}).value||"";
    q=q.trim().toLowerCase();
    var sortMode=(document.getElementById("discsort")||{}).value||"name-asc";
    names.forEach(function(name){
      var res=pv[name]||{};
      var models=(res.models||[]).slice();
      if(q) models=models.filter(function(m){ return m.toLowerCase().indexOf(q)>=0; });
      if(sortMode==="name-asc") models.sort();
      else if(sortMode==="name-desc") models.sort().reverse();
      else if(sortMode==="new-first"){ models.sort(); models.reverse(); models.sort(function(a,b){ return (discHave[b]?1:0)-(discHave[a]?1:0); }); }
      var hardErr=res.error && !(res.models&&res.models.length);
      // Hide a provider entirely when a filter excludes all of its models.
      if(q && !models.length && !hardErr) return;
      var det=document.createElement("details"); det.className="discprov"; det.open=true;
      var sum=document.createElement("summary");
      var total=(res.models||[]).length;
      var label=q?(models.length+"/"+total):(""+total);
      var meta;
      if(hardErr){ meta='<span class="pvmeta pverr">error: '+esc(res.error)+'</span>'; }
      else {
        meta='<span class="pvmeta">'+label+' model(s)</span>';
        if(res.stale) meta+=' <span class="pvmeta pvwarn">⚠ cached (refresh failed: '+esc(res.error||"")+')</span>';
        else if(res.fallback==="static") meta+=' <span class="pvmeta pvwarn">⚠ static list (live discovery failed: '+esc(res.error||"")+')</span>';
      }
      sum.innerHTML=(ICONS[name]||"")+" <b>"+esc(name)+"</b> "+meta;
      det.appendChild(sum);
      var body=document.createElement("div"); body.className="discbody";
      var pAll=addAllBtn(models,"addall"," shown"); if(pAll) body.appendChild(pAll);
      // Group by family. Skip the nesting when there'd only be one group.
      var keyOf=groupKeyer(models);
      var groups={}, order=[];
      models.forEach(function(m){ var g=keyOf(m); if(!groups[g]){ groups[g]=[]; order.push(g); } groups[g].push(m); });
      if(order.length<=1){
        models.forEach(function(m){ body.appendChild(mdlChip(m)); });
      } else {
        order.sort();
        order.forEach(function(g){
          var list=groups[g];
          var gd=document.createElement("details"); gd.className="discgrp"; gd.open=!!q;
          var gs=document.createElement("summary");
          var addableN=list.filter(function(m){ return !discHave[m]; }).length;
          gs.innerHTML='<b>'+esc(g)+'</b> <span class="pvmeta">'+list.length+(addableN?"":" · all added")+'</span>';
          gd.appendChild(gs);
          var gb=document.createElement("div"); gb.className="discgrpbody";
          var gAll=addAllBtn(list,"addall"," in "+esc(g)); if(gAll) gb.appendChild(gAll);
          list.forEach(function(m){ gb.appendChild(mdlChip(m)); });
          gd.appendChild(gb); body.appendChild(gd);
        });
      }
      det.appendChild(body);
      box.appendChild(det);
    });
  }

  var db=document.getElementById("discbtn");
  if(db) db.addEventListener("click",function(){
    var box=document.getElementById("discresult");
    box.textContent="querying providers…";
    var tb=document.getElementById("disctools"); if(tb) tb.style.display="none";
    fetch("/_models/available").then(function(r){return r.json();}).then(function(d){
      discData=d; discHave={}; (d.in_ladder||[]).forEach(function(m){discHave[m]=1;});
      if(tb){ tb.style.display=""; }
      renderDisc();
    }).catch(function(){ box.textContent="discovery failed"; });
  });
  var df=document.getElementById("discfilter"); if(df) df.addEventListener("input",renderDisc);
  var ds=document.getElementById("discsort"); if(ds) ds.addEventListener("change",renderDisc);

  load();
})();
