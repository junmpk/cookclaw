"use strict";
const $ = (id) => document.getElementById(id);
const names = {research:"食谱研究",diet:"饮食分析",menu:"菜单规划",system:"工作流"};
const labels = {running:"协作进行中",awaiting_confirmation:"等待确认",completed:"模拟执行完成",cancelled:"已取消",blocked:"需要调整需求",error:"执行未完成",interrupted:"等待恢复"};
let run = null, stream = null, cursor = 0, eventCount = 0, modelCalls = 0, tokens = 0, replaying = false;
let streamGeneration = 0;
let chatBusy = false;
let chatThread = localStorage.getItem("cookclaw-chat-thread") || null;
let chatHistory = [];
try { chatHistory = JSON.parse(localStorage.getItem("cookclaw-chat-history") || "[]"); } catch {}
if(!Array.isArray(chatHistory)) chatHistory=[];
const presets = {
  family:{request:"给三口之家安排三菜一汤，偏清淡，尽量利用鸡肉、西兰花和鸡蛋。说明搭配取舍和还需要核对的信息。",people:3,dishes:3,soups:1,exclusions:"花生"},
  exclusion:{request:"两个人吃晚饭，两菜一汤。优先选择蔬菜和鸡肉，说明排除食材检查的依据和未知项。",people:2,dishes:2,soups:1,exclusions:"花生,鸡蛋"},
  shortage:{request:"需要五道不重复的菜和两道汤。如果候选不足，请补查后如实说明，不能编造食谱。",people:4,dishes:5,soups:2,exclusions:"花生,鸡蛋,牛奶,小麦,大豆"}
};
function node(tag, text, className) { const el = document.createElement(tag); if(text !== undefined) el.textContent = text; if(className) el.className = className; return el; }
function showNotice(text) { $("notice").textContent=text; $("notice").hidden=!text; }
async function api(path, body) {
  const response=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw new Error(typeof data.detail==="string"?data.detail:"输入不符合要求，请检查数量和需求。");
  return data;
}
function brief() { return {request:$("request").value.trim(),people:Number($("people").value),dishes:Number($("dishes").value),soups:Number($("soups").value),exclusions:$("exclusions").value.split(/[,，、]/).map(x=>x.trim()).filter(Boolean),mode:$("mode").value}; }
function fill(data) { for(const key of ["request","people","dishes","soups","mode"]) if(data[key]!==undefined) $(key).value=data[key]; $("exclusions").value=Array.isArray(data.exclusions)?data.exclusions.join(","):data.exclusions; modeNote(); }
function modeNote() { $("mode-note").textContent=$("mode").value==="live"?"真实调用三个 LangChain Agent；会消耗模型额度。食谱仅来自真实检索。":"规则驱动，沿同一张 LangGraph 图运行；不调用模型。"; }
$("mode").addEventListener("change",modeNote);
document.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{ fill(presets[button.dataset.preset]); document.querySelectorAll("[data-preset]").forEach(b=>b.classList.toggle("selected",b===button)); }));

function chatTime() { return new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}); }
function addChatBubble(text, direction="incoming", options={}) {
  const bubble=node("article",undefined,"chat-bubble "+(direction==="outgoing"?"outgoing":"incoming")+(options.typing?" typing":""));
  if(options.typing) bubble.append(node("span","正在输入"));
  else text.split(/\n+/).filter(Boolean).forEach(line=>bubble.append(node("p",line)));
  bubble.append(node("time",options.time||chatTime()));
  $("chat-messages").append(bubble);
  $("chat-messages").scrollTop=$("chat-messages").scrollHeight;
  return bubble;
}
function addChatAction(label, action) {
  const wrapper=node("div",undefined,"chat-quick-replies");
  const button=node("button",label);button.type="button";button.dataset.chatAction=action;wrapper.append(button);$("chat-messages").append(wrapper);$("chat-messages").scrollTop=$("chat-messages").scrollHeight;
}
function saveChat(text, role) {
  chatHistory.push({text,role});chatHistory=chatHistory.slice(-40);
  localStorage.setItem("cookclaw-chat-history",JSON.stringify(chatHistory));
}
function renderChatResult(data) {
  stream?.close();streamGeneration++;
  $("main-title").textContent="当前厨房任务";
  $("status").textContent="已回复";
  $("status").className="status completed";
  $("run-meta").textContent="共享对话服务 · "+data.response.handled_by;
  $("request-summary").hidden=true;
  document.querySelector(".agents-grid").hidden=true;
  document.querySelector(".flow").hidden=true;
  $("results").hidden=true;$("empty").hidden=true;
  $("chat-result").hidden=false;
  $("chat-answer").textContent=data.reply || "请查看当前任务。";
  const candidates=data.response.data?.recipes || data.state.candidate_recipes || [];
  $("chat-recipes").replaceChildren();
  candidates.forEach((r,i)=>{
    const card=recipeCard({id:r.id,name:r.name,source:r.recipe_detail?.source||"",image_url:r.image,
      ingredients:r.ingredients||[],nutrition:{},kind:r.tags?.includes("soup")?"soup":"dish"});
    const button=node("button","查看第 "+(i+1)+" 道做法","secondary");button.type="button";
    button.addEventListener("click",()=>sendChatMessage("详情 "+(i+1)));
    card.append(button);$("chat-recipes").append(card);
  });
  $("chat-state").textContent=JSON.stringify(data.state,null,2);
  $("trace").replaceChildren();
  const traces=data.traces||[];
  $("calls").textContent=traces.reduce((s,t)=>s+(t.model_call_count||0),0);
  $("tokens").textContent=traces.reduce((s,t)=>s+(t.token_usage?.total_tokens||0),0);
  $("rounds").textContent="—";
  const events=traces.flatMap(t=>t.events||[]);$("event-count").textContent=events.length;
  for(const event of events){const item=node("details",undefined,"trace-item");item.append(node("summary",event.kind||"回合事件"),node("pre",JSON.stringify(event,null,2)));$("trace").append(item);}
  $("replay").disabled=true;$("retry").hidden=true;
}
async function pollGraphRun(runId) {
  for(let attempt=0;attempt<240;attempt++) {
    try {
      const row=await api("/api/runs/"+runId);
      if(row.status!=="running") {
        run=row;resetTrace();renderRun();connect();return;
      }
      $("status").textContent="三 Agent 协作中";
      $("status").className="status running";
      $("run-meta").textContent="LangGraph · 研究 → 分析 → 规划 → 校验";
    } catch(error) { showNotice("协作状态读取失败："+error.message);return; }
    await new Promise(resolve=>setTimeout(resolve,750));
  }
  showNotice("协作等待时间较长，可稍后刷新查看任务状态。");
}
async function sendChatMessage(message) {
  const text=message.trim();
  if(!text||chatBusy)return;
  addChatBubble(text,"outgoing");
  saveChat(text,"outgoing");
  $("chat-input").value="";
  chatBusy=true;$("chat-send").disabled=true;
  const typing=addChatBubble("","incoming",{typing:true});
  try {
    const response=await api("/api/chat",{message:text,thread_id:chatThread});
    chatThread=response.thread_id;localStorage.setItem("cookclaw-chat-thread",chatThread);
    typing.remove();addChatBubble(response.reply,"incoming");
    saveChat(response.reply,"incoming");renderChatResult(response);
    if(response.graph_run_id) pollGraphRun(response.graph_run_id);
  } catch(error) { typing.remove();addChatBubble("暂时没有收到回复："+error.message,"incoming"); }
  finally { chatBusy=false;$("chat-send").disabled=false; }
}
$("chat-form").addEventListener("submit",event=>{event.preventDefault();sendChatMessage($("chat-input").value);});
$("chat-input").addEventListener("keydown",event=>{if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();$("chat-form").requestSubmit();}});
chatHistory.forEach(x=>{if(typeof x.text==="string")addChatBubble(x.text,x.role);});
$("new-chat").addEventListener("click",()=>{
  if(chatBusy)return;
  const quickReplies=$("chat-messages").querySelector(".chat-quick-replies");
  chatThread=null;chatHistory=[];run=null;stream?.close();stream=null;
  localStorage.removeItem("cookclaw-chat-thread");localStorage.removeItem("cookclaw-chat-history");
  $("chat-messages").replaceChildren();addChatBubble("新的对话开始了。今天想解决什么厨房问题？");
  if(quickReplies)$("chat-messages").append(quickReplies);
  resetTrace();$("chat-result").hidden=true;$("results").hidden=true;$("empty").hidden=false;
  document.querySelector(".agents-grid").hidden=true;document.querySelector(".flow").hidden=true;
  $("status").textContent="等待开始";$("status").className="status";
  $("run-meta").textContent="自然语言 → 共享回合服务 → 工具与会话状态 → 回复";
});
document.querySelectorAll("[data-chat-message]").forEach(button=>button.addEventListener("click",()=>sendChatMessage(button.dataset.chatMessage)));
$("chat-messages").addEventListener("click",event=>{
  const action=event.target.closest("[data-chat-action]")?.dataset.chatAction;
  if(action==="focus-brief"){$("brief-settings").open=true;$("request").focus();}
});
function resetTrace() { cursor=eventCount=modelCalls=tokens=0; $("trace").replaceChildren(); $("calls").textContent="0";$("tokens").textContent="—";$("rounds").textContent="0";$("event-count").textContent="0";document.querySelectorAll(".agent-card").forEach(el=>{el.classList.remove("active");el.querySelector(".agent-state").textContent="待命";});document.querySelectorAll(".flow .active").forEach(el=>el.classList.remove("active")); }
function eventTitle(event) {
  const d=event.data, role=names[event.role]||event.role;
  const nodeNames={research:"检索候选",dietary:"分析饮食要求",menu:"组合菜单",validate:"规则校验",review:"审阅菜单",refetch:"补查候选",execute:"Mock 执行",blocked:"停止修订"};
  if(event.kind==="node_start") return role+" · "+(nodeNames[d.node]||d.node);
  if(event.kind==="node_end") return (nodeNames[d.node]||d.node)+"完成 · "+d.duration_ms+" ms";
  if(event.kind==="tool_start") return "调用 "+d.tool;
  if(event.kind==="tool_end") return d.tool+(d.count!==undefined?" · "+d.count+" 条结果":" · 已返回");
  if(event.kind==="model_start") return role+" · 调用 "+d.model;
  if(event.kind==="model_end") return role+" · 模型返回";
  if(event.kind==="run_start") return d.mode==="live"?"真实模型协作开始":"规则演练开始（无模型调用）";
  if(event.kind==="run_end") return labels[d.status]||d.status;
  return d.message||event.kind;
}
function renderEvent(event) {
  if(event.id<=cursor)return; cursor=event.id;eventCount++;
  const item=node("article",undefined,"trace-item "+event.role+(event.kind==="error"?" error":""));
  item.append(node("h4",eventTitle(event)),node("div","v"+event.version+" · "+new Date(event.created*1000).toLocaleTimeString(),"time"));
  const details=node("details");details.append(node("summary","查看结构化记录"),node("pre",JSON.stringify(event.data,null,2)));item.append(details);$("trace").append(item);
  $("event-count").textContent=eventCount;
  if(event.kind==="model_start")$("calls").textContent=++modelCalls;
  if(event.kind==="model_end"){tokens+=(event.data.input_tokens||0)+(event.data.output_tokens||0);$("tokens").textContent=tokens||"未返回";}
  if(!run||event.version!==run.version)return;
  const card=$("agent-"+event.role);
  if(card&&["node_start","node_end"].includes(event.kind)){card.classList.toggle("active",event.kind==="node_start");card.querySelector(".agent-state").textContent=event.kind==="node_start"?"进行中":"已完成";}
  if(event.kind==="node_start"){document.querySelectorAll(".flow .active").forEach(el=>el.classList.remove("active"));document.querySelector(`[data-node="${event.data.node}"]`)?.classList.add("active");}
  if(event.kind==="node_end") {
    const output=event.data.output||{};
    run.state={...run.state,...output};
    $("rounds").textContent=run.state.revision_count||0;
    if(output.menu||output.review||output.dietary||output.issues)renderResult();
  }
  if(event.kind==="error")showNotice(event.data.message);
}
function recipeImageUrl(recipe) {
  if(recipe.image_url) return recipe.image_url;
  if(recipe.source?.includes("myplate.food/recipes/")) {
    return "https://storage.googleapis.com/peppermint-cdn/myplate.food-recipe-images/"+encodeURIComponent(recipe.id)+".jpg";
  }
  return "";
}
function recipeCard(recipe) {
  const card=node("article",undefined,"recipe");
  const imageUrl=recipeImageUrl(recipe);
  const imageWrap=node("div",undefined,"recipe-image");
  if(imageUrl){
    const image=node("img");image.src=imageUrl;image.alt=recipe.name;image.loading="lazy";image.decoding="async";
    image.addEventListener("error",()=>{image.remove();imageWrap.classList.add("image-missing");imageWrap.append(node("span","图片暂不可用"));},{once:true});imageWrap.append(image);
  } else { imageWrap.classList.add("image-missing");imageWrap.append(node("span","暂无公开图片")); }
  card.append(imageWrap);
  const head=node("div",undefined,"recipe-title");
  head.append(node("div",recipe.kind==="soup"?"SOUP / 汤":"DISH / 菜","recipe-kind"),node("h3",recipe.name));
  try{const url=new URL(recipe.source);if(url.protocol==="https:"){const source=node("a","查看原始食谱 ↗","source");source.href=url.href;source.target="_blank";source.rel="noopener noreferrer";head.append(source);}}catch{}
  card.append(head);const details=node("details");details.append(node("summary","食材与来源营养字段"));const list=node("ul");recipe.ingredients.forEach(x=>list.append(node("li",x)));details.append(list);
  if(Object.keys(recipe.nutrition||{}).length){details.append(node("p","以下是原食谱提供的数据，未按家庭人数或实际食用量换算。","nutrition-note"));const facts=node("ul");Object.entries(recipe.nutrition).forEach(([k,v])=>facts.append(node("li",k+": "+v)));details.append(facts);}
  card.append(details);return card;
}
function renderResult() {
  const state=run.state||{}, ids=state.menu?.recipe_ids||[], recipes=state.candidates||[];
  $("results").hidden=!state.menu&&!state.dietary;$("empty").hidden=!$("results").hidden;
  $("recipes").replaceChildren();ids.forEach(id=>{const r=recipes.find(r=>r.id===id);if(r)$("recipes").append(recipeCard(r));});
  $("explanation").textContent=state.menu?.explanation||"";$("menu-count").textContent=ids.length?ids.length+" 道候选":"";
  $("analysis").replaceChildren();[...(state.dietary?.advice||[]),...(state.review?.findings||[]),...(state.dietary?.unknowns||[]),...(state.review?.unknowns||[])].filter((v,i,a)=>a.indexOf(v)===i).forEach(text=>$("analysis").append(node("li",text)));
  $("rule-links").replaceChildren();if(state.dietary){const a=node("a","饮食原则来源：WHO ↗");a.href="https://www.who.int/news-room/fact-sheets/detail/healthy-diet";a.target="_blank";a.rel="noopener noreferrer";$("rule-links").append(a,node("span","食材排除与证据要求：项目规则"));}
  $("issues").replaceChildren();(state.issues||[]).forEach(text=>$("issues").append(node("li",text)));$("issues-box").hidden=!(state.issues||[]).length;
  $("approval-box").hidden=run.status!=="awaiting_confirmation"||replaying;
  $("execution").hidden=!state.execution;$("execution").textContent=state.execution?state.execution.message+"\n编号："+state.execution.execution_id:"";
}
function renderRun() {
  $("chat-result").hidden=true;document.querySelector(".agents-grid").hidden=false;document.querySelector(".flow").hidden=false;
  $("status").textContent=(replaying?"历史记录 · ":"")+(labels[run.status]||run.status);$("status").className="status "+run.status;
  $("run-meta").textContent="方案 v"+run.version+" · "+(run.brief.mode==="live"?"真实模型协作":"流程演练 / 规则驱动")+" · "+run.id.slice(0,8);
  $("request-summary").hidden=false;$("request-summary").textContent=run.brief.request+"\n"+run.brief.people+" 人 · "+run.brief.dishes+" 菜 "+run.brief.soups+" 汤 · 排除："+(run.brief.exclusions.join("、")||"未设置");
  const busy=run.status==="running";$("start").disabled=busy;$("revise").disabled=busy||replaying;$("replay").disabled=busy;$("retry").hidden=!["error","interrupted"].includes(run.status)||replaying;
  $("approve").disabled=busy;$("reject").disabled=busy;
  if(run.status==="awaiting_confirmation")document.querySelector('[data-node="approval"]').classList.add("active");
  renderResult();
}
async function refresh() { if(run){run=await api("/api/runs/"+run.id);renderRun();} }
function connect() {
  stream?.close();const generation=++streamGeneration;
  stream=new EventSource(`/api/runs/${run.id}/events?after=${cursor}`);
  stream.onmessage=event=>{if(generation===streamGeneration)renderEvent(JSON.parse(event.data));};
  stream.addEventListener("done",async()=>{if(generation!==streamGeneration)return;stream.close();await refresh();});
  stream.onerror=()=>{stream.close();if(generation===streamGeneration)setTimeout(()=>{if(generation===streamGeneration)connect();},2000);};
}
async function perform(action) {try{showNotice("");await action();}catch(error){showNotice(error.message);if(run)await refresh().catch(()=>{});else $("start").disabled=false;}}
$("brief-form").addEventListener("submit",event=>{event.preventDefault();perform(async()=>{ $("start").disabled=true;const data=await api("/api/runs",brief());stream?.close();replaying=false;resetTrace();run=await api("/api/runs/"+data.id);localStorage.setItem("cookclaw-demo-run",data.id);renderRun();connect();});});
$("revise").addEventListener("click",()=>perform(async()=>{if(!$("brief-form").reportValidity())return;await api(`/api/runs/${run.id}/revise`,{version:run.version,brief:brief()});replaying=false;await refresh();connect();}));
for(const [id,approved] of [["approve",true],["reject",false]])$(id).addEventListener("click",()=>perform(async()=>{$("approve").disabled=$("reject").disabled=true;await api(`/api/runs/${run.id}/decision`,{version:run.version,approved});await refresh();connect();}));
$("retry").addEventListener("click",()=>perform(async()=>{await api(`/api/runs/${run.id}/retry`,{});await refresh();connect();}));
$("replay").addEventListener("click",()=>perform(async()=>{replaying=!replaying;$("replay").textContent=replaying?"返回当前方案":"查看本次已保存记录";resetTrace();await refresh();connect();}));
(async()=>{try{
  const config=await api("/api/config");$("live-option").disabled=!config.live_available;
  if(!config.recipes_ready)showNotice("食谱库尚未准备，请运行：uv run python -m app.demo.seed");
  else if(!config.live_available)showNotice("自然语言聊天需要配置 DASHSCOPE_API_KEY；高级面板可运行独立的规则演练。");
  $("main-title").textContent="当前厨房任务";
  $("run-meta").textContent="自然语言 → 共享回合服务 → 工具与会话状态 → 回复";
  document.querySelector(".agents-grid").hidden=true;document.querySelector(".flow").hidden=true;
  $("empty").querySelector("h3").textContent="从一句话开始，完成今天这一餐";
  $("empty").querySelector("p").textContent="在左侧聊聊食材、人数和忌口，也可以问厨房技巧。这里会展示本轮回答与真实食谱。";
  if(chatHistory.length){$("chat-answer").textContent=chatHistory.filter(x=>x.role==="incoming").at(-1)?.text||"";$("chat-result").hidden=false;$("empty").hidden=true;}
}catch(error){showNotice("服务暂不可用："+error.message);}})();
