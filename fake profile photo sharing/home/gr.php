<?php
include("dbconnect.php");
session_start();
extract($_POST);
$fid = $_REQUEST['id'];
$upid = $_REQUEST['upid'];
$gname = $_REQUEST['gname'];
$act = $_REQUEST['act'];

$accept_count_query = mysql_query("SELECT COUNT(*) AS accept_count FROM grpost WHERE gname='$gname' AND status='1' AND pid='$fid'");
$accept_count_result = mysql_fetch_assoc($accept_count_query);
$accept_count = $accept_count_result['accept_count'];
$reject_count_query = mysql_query("SELECT COUNT(*) AS reject_count FROM grpost WHERE gname='$gname' AND status='2' AND pid='$fid'");
$reject_count_result = mysql_fetch_assoc($reject_count_query);
$reject_count = $reject_count_result['reject_count'];
$ghj=mysql_query("select * from grouplist where gname='$gname'");
$group_size=mysql_num_rows($ghj);


$python_script = "python predict.py $group_size $accept_count $reject_count";
$prediction = shell_exec($python_script);

if ($prediction<1) {
    $qr=mysql_query("update gpost set status='1' where id='$fid'");

    echo $mobile=$p7;

   $msg='Your post has been accepted by your group members';
   
   
   ?>
   <iframe src="http://sms.creativepoint.in/api/push.json?apikey=6555c521622c1&route=transsms&sender=FSSMSS&mobileno=<?php echo $mobile;?>&text=Dear customer your msg is <?php echo $msg;?>  Sent By FSMSG FSSMSS" style="display:sucess" class="iframe"></iframe>
   
   
   
   
    
      
      <?php
} else {
    $qr=mysql_query("update gpost set status='2' where id='$fid'");

    echo  $mobile=$p7;
    
    $msg='Your post has been rejected by your group members';
    
    
    ?>
    <iframe src="http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=<?php echo $mobile;?>&from=FSSMSS&message=Dear User your msg is <?php echo $msg;?> Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640" style="display:sucess" class="iframe"></iframe>
    
    
    
    
     
       
       <?php
}
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Facebook Home</title>
	<link href="css/bootstrap.min.css" rel="stylesheet">
	<link href="css/font-awesome.min.css" rel="stylesheet">
	<link href="css/datepicker3.css" rel="stylesheet">
	<link href="css/styles.css" rel="stylesheet">
	
	<!--Custom Font-->
	<link href="https://fonts.googleapis.com/css?family=Montserrat:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">
	<!--[if lt IE 9]>
	<script src="js/html5shiv.js"></script>
	<script src="js/respond.min.js"></script>
	<![endif]-->
    <style type="text/css">
<!--
.im {border-radius: 50%;
}
.im1 {  border-radius: 50%;
}
.im2 {  border-radius: 50%;
}
.style1 {color: #000000}
-->
    </style>
</head>
<body>
	<nav class="navbar navbar-custom navbar-fixed-top" role="navigation">
		<div class="container-fluid">
			<div class="navbar-header">
				<button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#sidebar-collapse"><span class="sr-only">Toggle navigation</span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span>
					<span class="icon-bar"></span></button>
				<a class="navbar-brand" href="#"><span>Facebook</span>Home</a>
				<ul class="nav navbar-top-links navbar-right">
				</ul>
			</div>
		</div><!-- /.container-fluid -->
	</nav>
	<div id="sidebar-collapse" class="col-sm-3 col-lg-2 sidebar">
		<div class="profile-sidebar">
			<div class="profile-userpic">
			<?php
					$qry=mysql_query("select * from register where id='$uid'");
					$rw=mysql_fetch_array($qry); 
					?>
				
			</div>
			<div class="profile-usertitle">
				<div class="profile-usertitle-name"><?php echo $rw['name'];?></div>
				<div class="profile-usertitle-status"><span class="indicator label-success"></span>&nbsp;&nbsp;&nbsp;&nbsp;<?php echo $rw['name'];?></div>
			</div>
			<div class="clear"></div>
		</div>
		<div class="divider"></div>
		<form role="search">
			<div class="form-group">
				<input type="text" class="form-control" placeholder="Search">
			</div>
		</form>
		<ul class="nav menu">
			<li><a href="home.php"><i class="fa fa-home"></i> <span>Home</span></a></li>
										 <li><a href="frlist.php"><i class="fa fa-book"></i> <span>Friend List</span></a></li>
										<li><a href="list.php"><i class="fa fa-users"></i> <span>List</span></a></li>
									<li><a href="post.php"><i class=" fa fa-book"></i> <span>Post</span></a></li>
									<li><a href="accept.php"><i class="fa fa-users"></i> <span>Accept</span></a></li>
									<li><a href="group.php"><i class="fa fa-users"></i> <span>Create Group</span></a></li>
									<li><a href="grouppost.php"><i class="fa fa-users"></i> <span>Group Post</span></a></li>
									
									<li><a href="profile.php"><i class="fa fa-bell-o"></i> <span>Profile</span></a></li>
									<li><a href="index.php"><i class="fa fa-power-switch"></i> <span>Logout</span></a></li>
		</ul>
	</div><!--/.sidebar-->
		
	<div class="col-sm-9 col-sm-offset-3 col-lg-10 col-lg-offset-2 main">
		<div class="row">
			<ol class="breadcrumb">
				<li><a href="#">
					<em class="fa fa-home"></em>
				</a></li>
				<li class="active">Forms</li>
			</ol>
		</div><!--/.row-->
		
		<div class="row">
			<div class="col-lg-12">
				<h1 class="page-header">Post List </h1>
			</div>
		</div><!--/.row-->
				
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading">List</div>
				  <div class="panel-body">
				  
				  

				  <?php 
				  
				  
		
																	$qr=mysql_query("select * from grpost where frid='$uid' && status='0'");
																	
																	while($row=mysql_fetch_array($qr))
																	{
																	echo $fid=$row['uid'];
																
																	echo $pid=$row['pid'];
																	
																	
																	$qt=mysql_query("select * from gpost where  id='$pid'  ORDER BY id DESC");
																	
																	
																	
																	
																	$r=mysql_fetch_array($qt);
																	
																	$rt=$r['sid'];
																	$g=mysql_query("select * from register where id='$rt'");
																	$ghh=mysql_fetch_array($g);
																	
																	
																	?>
          </h3>				  
				    <form action="" method="post" name="form">
                                                    <table width="91%" border="0">
                                                
                                                      <tr>
                                                        <td height="39">&nbsp;</td>
                                                        <td colspan="3"><span class="style2"><img src="images/<?php echo $ghh['image'];?>" width="25" height="25" class="im1">&nbsp;<?php echo $ghh['name'];?>&nbsp;&nbsp;&nbsp;<?php echo $r['type'];?></span></td>
                                                        <td></td>
                                                      </tr>
                                                      <tr>
                                                        <td height="32">&nbsp;</td>
                                                        <td colspan="3">&nbsp;</td>
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                      <tr>
                                                        <td height="32">&nbsp;</td>
                                                        <td colspan="3"><div align="left" class="task-desc task-desc"><?php echo $r['cption'];?></div></td>
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                      <tr>
                                                        <td width="16%" height="224">&nbsp;</td>
                                                        <td colspan="3" valign="top"><div align="left"><img src="image/<?php echo $r['image'];?>" width="250" height="200" ></div></td>
                                                        <td width="30%">&nbsp;</td>
                                                      </tr>
                                                      <tr>
                                                        <td>&nbsp;</td>
                                                        <td width="25%"><div align="center" class="task-desc task-desc"><a href="grouppost.php?id=<?php echo $pid;?>&act=snt&upid=<?php echo $row['uid'];?>&gname=<?php echo $row['gname'];?>">Accept</a></div></td>
                                                        <td width="19%"><div align="left" class="task-desc task-desc"><a href="grouppost.php?id=<?php echo $pid;?>&act=rjt&upid=<?php echo $row['uid'];?>&gname=<?php echo $row['gname'];?>">Reject</a></div></td>
                                                        <td width="10%"><div align="left" class="task-desc task-desc"><a href="share.php?id=<?php echo $r['id'];?>"></a></div></td>
                                                        <td>&nbsp;</td>
                                                      </tr>
                                                    </table>
                                                    <?php
													
													}
																																	
																	?>
                                                  </form>
				  </div>
				</div><!-- /.panel-->
			</div>
			<!-- /.panel-->
	  </div><!-- /.col-->
			<div class="col-sm-12">
				<p class="back-link">Lumino Theme by <a href="https://www.medialoot.com">Medialoot</a></p>
			</div>
</div><!-- /.row -->
	</div><!--/.main-->
	
<script src="js/jquery-1.11.1.min.js"></script>
	<script src="js/bootstrap.min.js"></script>
	<script src="js/chart.min.js"></script>
	<script src="js/chart-data.js"></script>
	<script src="js/easypiechart.js"></script>
	<script src="js/easypiechart-data.js"></script>
	<script src="js/bootstrap-datepicker.js"></script>
	<script src="js/custom.js"></script>
	
</body>
</html>
