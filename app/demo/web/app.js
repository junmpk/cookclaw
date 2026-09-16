"use strict";
const $ = (id) => document.getElementById(id);
const names = {research:"食谱研究",diet:"饮食分析",inventory:"库存分析",menu:"菜单规划",scheduler:"烹饪调度",system:"工作流"};
const labels = {running:"协作进行中",awaiting_confirmation:"等待确认",completed:"模拟执行完成",cancelled:"已取消",blocked:"需要调整需求",error:"执行未完成",interrupted:"等待恢复"};
let run = null, stream = null, cursor = 0, eventCount = 0, modelCalls = 0, toolCalls = 0, tokens = 0, replaying = false;
let timelineEvents = [];
let streamGeneration = 0;
let pollGeneration = 0;
let chatBusy = false;
let chatThread = localStorage.getItem("cookclaw-chat-thread") || null;
let chatHistory = [];
try { chatHistory = JSON.parse(localStorage.getItem("cookclaw-chat-history") || "[]"); } catch {}
if(!Array.isArray(chatHistory)) chatHistory=[];
const presets = {
  family:{request:"给三口之家安排三菜一汤，偏清淡，尽量利用鸡肉、西兰花和鸡蛋，45分钟内做好。",people:3,dishes:3,soups:1,exclusions:"花生",availableIngredients:"鸡肉,西兰花,鸡蛋",budgetYuan:50,maxMinutes:45,equipment:"灶台,空气炸锅"},
  exclusion:{request:"两个人吃晚饭，两菜一汤。优先选择蔬菜和鸡肉，说明排除食材检查的依据和未知项。",people:2,dishes:2,soups:1,exclusions:"花生,鸡蛋",availableIngredients:"鸡肉,青菜",budgetYuan:"",maxMinutes:"",equipment:"灶台"},
  shortage:{request:"需要五道不重复的菜和两道汤。如果候选不足，请补查后如实说明，不能编造食谱。",people:4,dishes:5,soups:2,exclusions:"花生,鸡蛋,牛奶,小麦,大豆",availableIngredients:"",budgetYuan:"",maxMinutes:60,equipment:"灶台,烤箱"}
};
function node(tag, text, className) { const el = document.createElement(tag); if(text !== undefined) el.textContent = text; if(className) el.className = className; return el; }
function showNotice(text) { $("notice").textContent=text; $("notice").hidden=!text; }
async function api(path, body) {
  const response=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw new Error(typeof data.detail==="string"?data.detail:"输入不符合要求，请检查数量和需求。");
  return data;
}
function csvValue(id) { return $(id).value.split(/[,，、]/).map(x=>x.trim()).filter(Boolean); }
function optionalNumber(id) { const value=$(id).value.trim(); return value?Number(value):null; }
function brief() { return {request:$("request").value.trim(),people:Number($("people").value),dishes:Number($("dishes").value),soups:Number($("soups").value),exclusions:csvValue("exclusions"),available_ingredients:csvValue("available-ingredients"),budget_yuan:optionalNumber("budget-yuan"),max_minutes:optionalNumber("max-minutes"),equipment:csvValue("equipment"),mode:$("mode").value}; }
function fill(data) { for(const key of ["request","people","dishes","soups","mode"]) if(data[key]!==undefined) $(key).value=data[key]; $("exclusions").value=Array.isArray(data.exclusions)?data.exclusions.join(","):data.exclusions; $("available-ingredients").value=data.availableIngredients??"";$("budget-yuan").value=data.budgetYuan??"";$("max-minutes").value=data.maxMinutes??"";$("equipment").value=data.equipment??"";modeNote(); }
function modeNote() { $("mode-note").textContent=$("mode").value==="live"?"真实调用动态选择的 3～5 个 LangChain Agent；会消耗模型额度。食谱仅来自真实检索。":"规则驱动，沿动态 LangGraph 运行；不调用模型。"; }
$("mode").addEventListener("change",modeNote);
document.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{ fill(presets[button.dataset.preset]); document.querySelectorAll("[data-preset]").forEach(b=>b.classList.toggle("selected",b===button)); }));

function chatTime() { return new Date().toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}); }
function addChatBubble(text, direction="incoming", options={}) {
  const bubble=node("article",undefined,"chat-bubble "+(direction==="outgoing"?"outgoing":"incoming")+(options.typing?" typing":""));
  if(options.typing) bubble.append(node("span","正在输入"));
  else {
    if(options.imageUrl){const image=node("img");image.src=options.imageUrl;image.alt="用户上传的厨房图片";image.className="chat-uploaded-image";bubble.append(image);}
    text.split(/\n+/).filter(Boolean).forEach(line=>bubble.append(node("p",line)));
  }
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
const agentLabels={research:"食谱研究员",dietary:"饮食分析师",inventory:"食材库存师",menu:"菜单规划师",scheduler:"烹饪调度师"};
const dimensionLabels={menu:"菜单组合",dietary:"饮食约束",inventory:"库存采购",scheduling:"时间设备"};
function renderProfile(profile) {
  if(!profile){$("team-panel").hidden=true;return;}
  const selected=profile.selected_agents||[];
  $("team-panel").hidden=false;
  $("team-title").textContent="本轮启用 "+selected.length+" 个 Agent："+selected.map(x=>agentLabels[x]||x).join("、");
  $("complexity-level").textContent={simple:"简单",coordinated:"协作",advanced:"高级"}[profile.level]||profile.level;
  $("dimension-bars").replaceChildren();
  Object.entries(profile.dimensions||{}).forEach(([key,value])=>{
    const item=node("div",undefined,"dimension-item");
    const label=node("div");label.append(node("span",dimensionLabels[key]||key),node("strong",String(value)));
    const track=node("div",undefined,"dimension-track");const fill=node("i");fill.style.width=Math.min(100,Number(value)*25)+"%";track.append(fill);item.append(label,track);$("dimension-bars").append(item);
  });
  $("routing-reasons").replaceChildren();(profile.reasons||[]).forEach(reason=>$("routing-reasons").append(node("li",reason)));
  const cardIds={research:"research",dietary:"diet",inventory:"inventory",menu:"menu",scheduler:"scheduler"};
  Object.entries(cardIds).forEach(([agent,id])=>{
    const card=$("agent-"+id),enabled=selected.includes(agent);card.hidden=!enabled;card.classList.toggle("selected-agent",enabled);
    if(enabled&&card.querySelector(".agent-state").textContent==="按需加入")card.querySelector(".agent-state").textContent="已入队";
  });
  document.querySelectorAll("[data-optional-agent]").forEach(el=>el.hidden=!selected.includes(el.dataset.optionalAgent));
}
function renderSharedState() {
  const state=run?.state||{}, profile=state.complexity_profile||{};
  $("shared-state").replaceChildren();
  const values=[
    ["团队",(profile.selected_agents||[]).length?profile.selected_agents.length+" Agents":"待选择"],
    ["候选",(state.candidates||[]).length],
    ["菜单",(state.menu?.recipe_ids||[]).length],
    ["库存",state.inventory?"已分析":"—"],
    ["排期",state.schedule?"已生成":"—"],
    ["校验",state.plan_validation?.status||"—"],
    ["恢复",(run?.metrics?.attempt_count||1)>1?"第 "+run.metrics.attempt_count+" 次":"首次"],
    ["版本","v"+(run?.version||state.version||1)]
  ];
  values.forEach(([label,value])=>{const chip=node("span");chip.append(node("b",label),document.createTextNode(" "+value));$("shared-state").append(chip);});
}
function renderTimeline() {
  const completed=timelineEvents.filter(event=>event.kind==="node_end"&&event.data?.duration_ms!==undefined);
  $("execution-timeline").replaceChildren();
  if(!completed.length){$("execution-timeline").append(node("span","等待节点完成","muted"));return;}
  const starts=completed.map(event=>event.created-event.data.duration_ms/1000),ends=completed.map(event=>event.created);
  const min=Math.min(...starts),max=Math.max(...ends),span=Math.max(.1,max-min);
  completed.forEach(event=>{
    const row=node("div",undefined,"timeline-row");row.append(node("span",names[event.role]||event.role,"timeline-label"));
    const track=node("div",undefined,"timeline-track"),bar=node("i",event.data.node==="inventory_refresh"?"库存复核":(eventTitle(event).split("完成")[0]));
    bar.className="timeline-bar "+event.role;bar.style.left=((event.created-event.data.duration_ms/1000-min)/span*100)+"%";bar.style.width=Math.max(5,event.data.duration_ms/1000/span*100)+"%";track.append(bar);row.append(track);$("execution-timeline").append(row);
  });
}
function renderChatResult(data) {
  stream?.close();streamGeneration++;
  $("main-title").textContent="当前厨房任务";
  $("status").textContent="已回复";
  $("status").className="status completed";
  $("run-meta").textContent="共享对话服务 · "+data.response.handled_by;
  $("request-summary").hidden=true;
  const profile=data.response.data?.routing?.profile;
  document.querySelector(".agents-grid").hidden=!!profile?false:true;
  document.querySelector(".flow").hidden=!!profile?false:true;
  renderProfile(profile);
  $("results").hidden=true;$("empty").hidden=true;
  $("chat-result").hidden=false;
  $("chat-answer").textContent=data.reply || "请查看当前任务。";
  const candidates=data.response.data?.recipes || data.state.candidate_recipes || [];
  $("chat-recipes").replaceChildren();
  if(data.response.response_type==="image_ingredients_confirmation"){
    const confirmation=node("section",undefined,"ingredient-confirmation");
    confirmation.append(node("strong","请确认识别结果"));
    const chips=node("div",undefined,"ingredient-chips");
    (data.response.data?.ingredients||[]).forEach(x=>chips.append(node("span",x)));
    confirmation.append(chips,node("p","确认后才会把这些食材交给真实菜谱检索。"));
    $("chat-recipes").append(confirmation);
    addChatAction("✓ 就这些，开始检索","confirm-image");
    addChatAction("＋ 还要补充食材","supplement-image");
  }
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
  $("tool-calls").textContent=traces.reduce((s,t)=>s+(t.tool_call_count||0),0);
  $("tokens").textContent=traces.reduce((s,t)=>s+(t.token_usage?.total_tokens||0),0);
  $("rounds").textContent="—";
  $("duration").textContent="—";
  const events=traces.flatMap(t=>t.events||[]);$("event-count").textContent=events.length;
  for(const event of events){const item=node("details",undefined,"trace-item");item.append(node("summary",event.kind||"回合事件"),node("pre",JSON.stringify(event,null,2)));$("trace").append(item);}
  $("replay").disabled=true;$("retry").hidden=true;
}
async function pollGraphRun(runId) {
  const generation=++pollGeneration;
  localStorage.setItem("cookclaw-demo-run",runId);
  for(let attempt=0;attempt<240;attempt++) {
    if(generation!==pollGeneration)return;
    try {
      const row=await api("/api/runs/"+runId);
      if(generation!==pollGeneration)return;
      if(row.status!=="running") {
        run=row;resetTrace();renderRun();connect();return;
      }
      $("status").textContent="多 Agent 协作中";
      $("status").className="status running";
      $("run-meta").textContent="LangGraph · 动态组队 → 并行分析 → 规划排期 → 校验";
    } catch(error) { if(generation===pollGeneration)showNotice("协作状态读取失败："+error.message);return; }
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
function readAsDataURL(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result||""));reader.onerror=()=>reject(new Error("图片读取失败"));reader.readAsDataURL(file);});}
async function sendChatImage(file){
  if(!file||chatBusy)return;
  if(!["image/jpeg","image/png","image/webp"].includes(file.type)){addChatBubble("只支持 JPEG、PNG 或 WebP 图片。","incoming");return;}
  if(file.size>8*1024*1024){addChatBubble("图片需要小于 8MB。","incoming");return;}
  chatBusy=true;$("chat-send").disabled=$("chat-attach").disabled=true;
  let dataUrl="";
  try{
    dataUrl=await readAsDataURL(file);
    const caption=$("chat-input").value.trim();$("chat-input").value="";
    addChatBubble(caption||"厨房图片","outgoing",{imageUrl:dataUrl});saveChat(caption||"[厨房图片]","outgoing");
    const typing=addChatBubble("","incoming",{typing:true});
    try{
      const response=await api("/api/chat/image",{data_url:dataUrl,caption,thread_id:chatThread});
      chatThread=response.thread_id;localStorage.setItem("cookclaw-chat-thread",chatThread);
      typing.remove();addChatBubble(response.reply,"incoming");saveChat(response.reply,"incoming");renderChatResult(response);
    }catch(error){typing.remove();addChatBubble("图片暂时没有处理完成："+error.message,"incoming");}
  }catch(error){addChatBubble(error.message||"图片读取失败","incoming");}
  finally{chatBusy=false;$("chat-send").disabled=$("chat-attach").disabled=false;$("chat-image").value="";}
}
$("chat-form").addEventListener("submit",event=>{event.preventDefault();sendChatMessage($("chat-input").value);});
$("chat-attach").addEventListener("click",()=>$("chat-image").click());
$("chat-image").addEventListener("change",()=>sendChatImage($("chat-image").files?.[0]));
$("chat-input").addEventListener("keydown",event=>{if(event.key==="Enter"&&!event.shiftKey&&!event.isComposing){event.preventDefault();$("chat-form").requestSubmit();}});
chatHistory.forEach(x=>{if(typeof x.text==="string")addChatBubble(x.text,x.role);});
$("new-chat").addEventListener("click",()=>{
  if(chatBusy)return;
  pollGeneration++;streamGeneration++;
  const quickReplies=$("chat-messages").querySelector(".chat-quick-replies");
  chatThread=null;chatHistory=[];run=null;stream?.close();stream=null;
  localStorage.removeItem("cookclaw-chat-thread");localStorage.removeItem("cookclaw-chat-history");
  localStorage.removeItem("cookclaw-demo-run");
  $("chat-messages").replaceChildren();addChatBubble("新的对话开始了。今天想解决什么厨房问题？");
  if(quickReplies)$("chat-messages").append(quickReplies);
  resetTrace();$("chat-result").hidden=true;$("results").hidden=true;$("empty").hidden=false;
  document.querySelector(".agents-grid").hidden=true;document.querySelector(".flow").hidden=true;
  $("team-panel").hidden=true;
  $("status").textContent="等待开始";$("status").className="status";
  $("run-meta").textContent="自然语言 → 共享回合服务 → 工具与会话状态 → 回复";
});
document.querySelectorAll("[data-chat-message]").forEach(button=>button.addEventListener("click",()=>sendChatMessage(button.dataset.chatMessage)));
$("chat-messages").addEventListener("click",event=>{
  const action=event.target.closest("[data-chat-action]")?.dataset.chatAction;
  if(action==="focus-brief"){$("brief-settings").open=true;$("request").focus();}
  if(action==="confirm-image")sendChatMessage("就这些");
  if(action==="supplement-image"){$("chat-input").placeholder="例如：再加豆腐和鸡蛋";$("chat-input").focus();}
});
function resetTrace() { cursor=eventCount=modelCalls=toolCalls=tokens=0;timelineEvents=[]; $("trace").replaceChildren(); $("calls").textContent="0";$("tool-calls").textContent="0";$("tokens").textContent="—";$("rounds").textContent="0";$("duration").textContent="—";$("event-count").textContent="0";$("execution-timeline").replaceChildren(node("span","执行后显示各 Agent 时间线","muted"));document.querySelectorAll(".agent-card").forEach(el=>{el.classList.remove("active");el.querySelector(".agent-state").textContent=el.classList.contains("optional-agent")?"已入队":"待命";});document.querySelectorAll(".flow .active").forEach(el=>el.classList.remove("active")); }
function eventTitle(event) {
  const d=event.data, role=names[event.role]||event.role;
  const nodeNames={research:"检索候选",dietary:"分析饮食要求",inventory:"分析库存覆盖",inventory_refresh:"复核库存覆盖",menu:"组合菜单",hydrate_details:"准备排期详情",schedule:"生成烹饪排期",validate:"菜单硬约束校验",review:"饮食证据审阅",final_validate:"跨 Agent 最终校验",refetch:"补查候选",execute:"Mock 执行",blocked:"停止修订"};
  if(event.kind==="node_start") return role+" · "+(nodeNames[d.node]||d.node);
  if(event.kind==="node_end") return (nodeNames[d.node]||d.node)+"完成 · "+d.duration_ms+" ms";
  if(event.kind==="tool_start") return "调用 "+d.tool;
  if(event.kind==="tool_end") return d.tool+(d.count!==undefined?" · "+d.count+" 条结果":" · 已返回");
  if(event.kind==="model_start") return role+" · 调用 "+d.model;
  if(event.kind==="model_end") return role+" · 模型返回";
  if(event.kind==="run_start") return d.mode==="live"?"真实模型协作开始":"规则演练开始（无模型调用）";
  if(event.kind==="recovery_start") return "从检查点恢复 · 第 "+d.attempt+" 次尝试";
  if(event.kind==="recovery_fallback") return "检查点尚未写入 · 安全重建初始输入";
  if(event.kind==="run_end") return labels[d.status]||d.status;
  return d.message||event.kind;
}
function renderEvent(event) {
  if(event.id<=cursor)return; cursor=event.id;eventCount++;
  const item=node("article",undefined,"trace-item "+event.role+(event.kind==="error"?" error":""));
  item.append(node("h4",eventTitle(event)),node("div","v"+event.version+" · "+new Date(event.created*1000).toLocaleTimeString(),"time"));
  const details=node("details");details.append(node("summary","查看结构化记录"),node("pre",JSON.stringify(event.data,null,2)));item.append(details);$("trace").append(item);
  $("event-count").textContent=eventCount;
  if(["node_start","node_end"].includes(event.kind)){timelineEvents.push(event);renderTimeline();}
  if(event.kind==="model_start")$("calls").textContent=++modelCalls;
  if(event.kind==="tool_start")$("tool-calls").textContent=++toolCalls;
  if(event.kind==="model_end"){tokens+=(event.data.input_tokens||0)+(event.data.output_tokens||0);$("tokens").textContent=tokens||"未返回";}
  if(!run||event.version!==run.version)return;
  const card=$("agent-"+event.role);
  if(card&&["node_start","node_end"].includes(event.kind)){card.classList.toggle("active",event.kind==="node_start");card.querySelector(".agent-state").textContent=event.kind==="node_start"?"进行中":"已完成";}
  if(event.kind==="node_start"){document.querySelectorAll(".flow .active").forEach(el=>el.classList.remove("active"));const aliases={inventory_refresh:"inventory",hydrate_details:"schedule",final_validate:"validate",review:"validate"};const flowNode=aliases[event.data.node]||event.data.node;document.querySelector(`[data-node="${flowNode}"]`)?.classList.add("active");}
  if(event.kind==="node_end") {
    const output=event.data.output||{};
    run.state={...run.state,...output};
    $("rounds").textContent=run.state.revision_count||0;
    if(output.menu||output.review||output.dietary||output.inventory||output.schedule||output.detail_hydration||output.plan_validation||output.issues)renderResult();
    renderSharedState();
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
  const inventory=state.inventory||{};$("inventory-box").hidden=!state.inventory;$("inventory-summary").replaceChildren();
  if(state.inventory){
    const availableText=(inventory.available_ingredients||[]).join("、");const available=node("p","已确认/提供食材："+(availableText||"尚未提供"));$("inventory-summary").append(available);
    if(inventory.explanation)$("inventory-summary").append(node("p",inventory.explanation,"nutrition-note"));
    const chosenCoverage=(inventory.candidate_coverage||[]).filter(item=>ids.includes(item.recipe_id));
    chosenCoverage.forEach(item=>{const recipe=recipes.find(r=>r.id===item.recipe_id);const preferred=(inventory.preferred_recipe_ids||[]).includes(item.recipe_id);const row=node("div",undefined,"coverage-row"+(preferred?" preferred":""));row.append(node("strong",(recipe?.name||item.recipe_id)+" · "+item.coverage_percent+"%"+(preferred?" · 库存优先":"")),node("span","已匹配 "+((item.matched_ingredients||[]).join("、")||"暂无")+"；缺口预览 "+((item.missing_preview||[]).join("、")||"无")));$("inventory-summary").append(row);});
    if((inventory.unknowns||[]).length)$("inventory-summary").append(node("p",inventory.unknowns.join(" "),"nutrition-note"));
  }
  const schedule=state.schedule||{};$("schedule-box").hidden=!state.schedule;$("schedule-list").replaceChildren();$("schedule-unknowns").replaceChildren();
  if(state.schedule){
    const hasDemo=(schedule.tasks||[]).some(task=>task.evidence_basis==="demo_template");$("schedule-total").textContent=schedule.total_minutes?(hasDemo?"演示排期约 ":"可核验时间约 ")+schedule.total_minutes+" 分钟":"部分时长待确认";
    (schedule.tasks||[]).forEach(task=>{const row=node("article",undefined,"schedule-task");row.append(node("span",String(task.order).padStart(2,"0"),"schedule-order"));const body=node("div");const basis=task.evidence_basis==="demo_template"?"演示模板":(task.evidence_basis==="source"?"来源步骤":"证据待补");const title=node("div",undefined,"schedule-title");title.append(node("strong",task.recipe_name),node("span",basis,"evidence-badge "+task.evidence_basis));body.append(title,node("p",(task.equipment||"待确认设备")+" · "+(task.start_minute===null||task.start_minute===undefined?"开始时间待确认":"第 "+task.start_minute+" 分钟开始")+" · "+(task.duration_minutes?task.duration_minutes+" 分钟":"时长待确认")));if(task.evidence)body.append(node("p",task.evidence,"schedule-evidence"));row.append(body);$("schedule-list").append(row);});
    (schedule.unknowns||[]).forEach(item=>$("schedule-unknowns").append(node("li",item)));
  }
  const validation=state.plan_validation||{};$("validation-box").hidden=!state.plan_validation;$("validation-items").replaceChildren();
  if(state.plan_validation){const statusText={ready:"校验通过",needs_confirmation:"有待确认项",blocked:"校验未通过"}[validation.status]||validation.status;$("validation-status").textContent=statusText;$("validation-status").className="validation-status "+validation.status;$("detail-notice").textContent=state.detail_hydration?.notice||"";[...(validation.blocking_issues||[]),...(validation.warnings||[])].forEach(item=>$("validation-items").append(node("li",item)));if(!(validation.blocking_issues||[]).length&&!(validation.warnings||[]).length)$("validation-items").append(node("li","菜单、库存与排期输出之间未发现冲突。"));}
  $("issues").replaceChildren();(state.issues||[]).forEach(text=>$("issues").append(node("li",text)));$("issues-box").hidden=!(state.issues||[]).length;
  $("approval-box").hidden=run.status!=="awaiting_confirmation"||replaying;
  $("execution").hidden=!state.execution;$("execution").textContent=state.execution?state.execution.message+"\n编号："+state.execution.execution_id:"";
}
function renderRun() {
  $("chat-result").hidden=true;document.querySelector(".agents-grid").hidden=false;document.querySelector(".flow").hidden=false;
  $("status").textContent=(replaying?"历史记录 · ":"")+(labels[run.status]||run.status);$("status").className="status "+run.status;
  const metrics=run.metrics||{};
  $("calls").textContent=metrics.model_call_count??modelCalls;$("tool-calls").textContent=metrics.tool_call_count??toolCalls;$("tokens").textContent=metrics.token_total||"—";$("rounds").textContent=metrics.node_count??0;$("duration").textContent=metrics.run_duration_ms===null||metrics.run_duration_ms===undefined?"—":metrics.run_duration_ms+" ms";
  $("run-meta").textContent="方案 v"+run.version+" · "+(run.brief.mode==="live"?"真实模型协作":"流程演练 / 规则驱动")+" · Run "+run.id.slice(0,8)+" · 尝试 "+(metrics.attempt_count||1);
  $("request-summary").hidden=false;$("request-summary").textContent=run.brief.request+"\n"+run.brief.people+" 人 · "+run.brief.dishes+" 菜 "+run.brief.soups+" 汤 · 排除："+(run.brief.exclusions.join("、")||"未设置")+" · 现有食材："+((run.brief.available_ingredients||[]).join("、")||"未设置")+" · 时限："+(run.brief.max_minutes?run.brief.max_minutes+" 分钟":"未设置");
  renderProfile(run.state?.complexity_profile);
  renderSharedState();
  const busy=run.status==="running";$("start").disabled=busy;$("revise").disabled=busy||replaying;$("replay").disabled=busy;$("retry").hidden=!["error","interrupted"].includes(run.status)||replaying;
  $("approve").disabled=busy;$("reject").disabled=busy;
  if(run.status==="awaiting_confirmation")document.querySelector('[data-node="approval"]').classList.add("active");
  renderResult();
}
async function refresh() { if(run){const id=run.id,generation=streamGeneration;const updated=await api("/api/runs/"+id);if(run?.id!==id||generation!==streamGeneration)return;run=updated;renderRun();} }
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
  $("rag-backend").textContent=config.recipe_backend==="hybrid_rag"?"Hybrid RAG · Dense + BM25 + RRF + Rerank":"公开离线演练检索";
  if(!config.recipes_ready)showNotice("食谱库尚未准备，请运行：uv run python -m app.demo.seed");
  else if(config.recipe_backend!=="hybrid_rag")showNotice("当前使用公开离线演练检索。要展示真实混合 RAG，请在 .env.demo 设置 DEMO_RECIPE_BACKEND=hybrid，并配置 RECIPE_MILVUS_URI。");
  else if(!config.live_available)showNotice("自然语言聊天需要配置 DASHSCOPE_API_KEY；高级面板可运行独立的规则演练。");
  $("main-title").textContent="当前厨房任务";
  $("run-meta").textContent="自然语言 → 共享回合服务 → 工具与会话状态 → 回复";
  document.querySelector(".agents-grid").hidden=true;document.querySelector(".flow").hidden=true;
  $("empty").querySelector("h3").textContent="从一句话开始，完成今天这一餐";
  $("empty").querySelector("p").textContent="在左侧聊聊食材、人数和忌口，也可以问厨房技巧。这里会展示本轮回答与真实食谱。";
  if(chatHistory.length){$("chat-answer").textContent=chatHistory.filter(x=>x.role==="incoming").at(-1)?.text||"";$("chat-result").hidden=false;$("empty").hidden=true;}
  const savedRun=localStorage.getItem("cookclaw-demo-run");
  if(savedRun){run=await api("/api/runs/"+encodeURIComponent(savedRun));resetTrace();renderRun();connect();}
}catch(error){showNotice("服务暂不可用："+error.message);}})();
