<?php
include("dbconnect.php");
session_start();
extract($_POST);
$uid=$_SESSION['uid'];
$fg=mysql_query("select * from register where id='$uid'");
$nm=mysql_fetch_array($fg);
$mn=$nm['name'];
$did=$_REQUEST['id'];
$at=$_REQUEST['act'];
if($at=='csnt')
{
$qry=mysql_query("update tag set status='2' where id='$did'");
$qry;
$qr=mysql_query("select * from tag where id='$did'");
$fg=mysql_fetch_array($qr);
$imid=$fg['imgid'];
$isid=$fg['uid'];
$dh=mysql_query("select * from upost where id='$imid'");
$kl=mysql_fetch_array($dh);
$fname=$kl['image'];
$caption=$kl['cption'];
$status=$kl['type'];
$tag='Tag With '.$mn;
$max_qry = mysql_query("select max(id) from upost");
	$max_row = mysql_fetch_array($max_qry); 
	$id=$max_row['max(id)']+1;
	$date=date("d-m-y");
	$t=date("h:i:sa");
	$qry1=mysql_query("insert into upost values('$id','$isid','$fname','$caption','$tag','','$date','$t')");
$qry1;
header("location:notification.php");
}
if($at=='crjt')
{
$qry=mysql_query("update tag set status='1' where id='$did'");
$qry;
header("location:notification.php");
}


if($at=='csnt1')
{
$qry=mysql_query("update comment set status='0' where id='$did'");
$qry;
header("location:notification.php");
}
if($at=='crjt1')
{
$qry=mysql_query("update comment set status='2' where id='$did'");
$qry;
header("location:notification.php");
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
									<li><a href="notification.php"><i class="fa fa-bell-o"></i> <span>Notification</span></a></li>
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
		  <div class="col-lg-12"></div>
		</div><!--/.row-->
				
		
		<div class="row">
			<div class="col-lg-12">
				<div class="panel panel-default">
					<div class="panel-heading"></div>
				  <div class="panel-body">
				   <form name="form1" method="post" action="">
                                  <div class="graph-visual tables-main">
                                    <h2 class="inner-tittle">Tag Post  </h2>
                                    <div class="graph">
                                      <div class="tables">
                                        <table class="table">
                                          <thead>
                                            <tr>
                                              <th>Sl.No</th>
                                              <th>Name</th>
                                              <th>Image</th>
                                              <th>Status</th>
                                            </tr>
                                          </thead>
                                          <?php
										  $i=1;
										  $qst=mysql_query("select * from tag where sid='$uid' && status='0'");
										  while($r=mysql_fetch_array($qst))
										  {
										 
										  
																	
									?>
                                          <tbody>
                                            <tr class="active">
                                              
                                              <th scope="row"><?php echo $i;?></th>
                                              <td valign="middle"><?php echo $r['name'];?></td>
											  <?php
											  $imid=$r['imgid'];
											  $qrt=mysql_query("select * from upost where id='$imid'");
											  $im=mysql_fetch_array($qrt);
											  ?>
                                              <td valign="middle"><img src="image/<?php echo $im['image'];?>" width="100" height="100"></td>
                                              <td valign="middle"><a href="notification.php?act=csnt&id=<?php echo $r['id'];?>">Accept</a>/<a href="notification.php?act=crjt&id=<?php echo $r['id'];?>">Reject</a></td>
                                            </tr>
                                          </tbody>
                                          <?php
										  $i++;
										  }
										  
										  ?>
                                        </table>
                                      </div>
                                    </div>
                                    <h3 class="inner-tittle two"><span class="inner-tittle">Comments</span></h3>
                                    <table class="table">
                                      <thead>
                                        <tr>
                                          <th>Sl.No</th>
                                          <th>Name</th>
                                          <th>Comments</th>
                                          <th>Status</th>
                                        </tr>
                                      </thead>
                                      <?php
										  $i=1;
										  $qst=mysql_query("select * from upost where sid='$uid'");
										  while($r=mysql_fetch_array($qst))
										  {
										  $mmid=$r['id'];
										  
									
									$qry=mysql_query("select * from comment where status='1' && mid='$mmid'");
									while($row=mysql_fetch_array($qry))
									{
									?>
                                      <tbody>
                                        <tr class="active">
                                          <?php
											$cid=$row['sid'];
											$rt=mysql_query("select * from register where id='$cid'");
											$r=mysql_fetch_array($rt);
											?>
                                          <th scope="row"><?php echo $i;?></th>
                                          <td valign="middle"><?php echo $r['name'];?></td>
                                          <td valign="middle"><?php echo $row['cmt'];?></td>
                                          <td valign="middle"><a href="notification.php?act=csnt1&id=<?php echo $row['id'];?>">Accept</a>/<a href="notification.php?act=crjt1&id=<?php echo $row['id'];?>">Reject</a></td>
                                        </tr>
                                      </tbody>
                                      <?php
										  $i++;
										  }
										  }
										  ?>
                                    </table>
                                    <p class="inner-tittle two">&nbsp;</p>
                                    <h2 class="inner-tittle">&nbsp;</h2>
                                    <p class="inner-tittle">&nbsp;</p>
                                  </div>
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
